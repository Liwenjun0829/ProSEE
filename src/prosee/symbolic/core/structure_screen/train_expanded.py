import json, logging, random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .bounded_model import BoundedStructureRegressor
from .encoding import TreeEncoder, canonical_structure
from .train import metrics, resolve_device


def clip_r2_score(score, config):
    clip_min = float(config.get("r2_clip_min", -10.0))
    clip_max = float(config.get("r2_clip_max", 1.0))
    if clip_min > clip_max:
        raise ValueError("r2_clip_min must be <= r2_clip_max")
    return float(np.clip(float(score), clip_min, clip_max))


def train_expanded(records, output_dir, config):
    seed = int(config["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=resolve_device(config); logging.info("NN training device=%s",device)
    unique = {}
    clipped_count = 0
    for row in records:
        try:
            score = float(row["estimated_r2"]); formula = row["formula"]
            if np.isfinite(score):
                clipped = clip_r2_score(score, config)
                clipped_count += int(clipped != score)
                unique.setdefault(canonical_structure(formula), (formula, clipped))
        except Exception: continue
    items = list(unique.values())
    if len(items) < 6: raise ValueError("expanded dataset requires at least 6 valid unique structures")
    np.random.default_rng(seed).shuffle(items); n=len(items); a=max(1,int(n*config["split"][0])); b=max(a+1,a+int(n*config["split"][1])); splits=[items[:a],items[a:b],items[b:]]
    print(
        "Neural-network data size: "
        f"expanded records={len(records)}, unique valid structures={n}, "
        f"clipped R²={clipped_count}, training set={len(splits[0])}, "
        f"validation set={len(splits[1])}, test set={len(splits[2])}"
    )
    enc=TreeEncoder(config["max_nodes"]).fit([f for f,_ in splits[0]])
    def ds(rows): return TensorDataset(torch.tensor([enc.encode(f) for f,_ in rows]),torch.tensor([v for _,v in rows],dtype=torch.float32))
    model=BoundedStructureRegressor(len(enc.vocab),config["embedding_dim"],config["hidden_dim"]).to(device); opt=torch.optim.Adam(model.parameters(),lr=config["learning_rate"]); loss_fn=torch.nn.MSELoss(); best=None; stale=0; history=[]
    total_epochs = int(config["epochs"])
    logging.info(
        "NN training started: train=%d validation=%d test=%d epochs=%d batch_size=%d",
        len(splits[0]), len(splits[1]), len(splits[2]), total_epochs,
        int(config["batch_size"]),
    )
    for epoch in range(total_epochs):
        model.train(); losses=[]
        for x,y in DataLoader(ds(splits[0]),batch_size=config["batch_size"],shuffle=True): x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): vx,vy=ds(splits[1]).tensors; vx,vy=vx.to(device),vy.to(device); vl=float(loss_fn(model(vx),vy))
        train_loss=float(np.mean(losses))
        history.append({"epoch":epoch+1,"train_loss":train_loss,"validation_loss":vl})
        improved = best is None or vl < best[0]
        if improved: best=(vl,{k:v.detach().clone() for k,v in model.state_dict().items()}); stale=0
        else: stale+=1
        logging.info(
            "NN epoch %d/%d | train_loss=%.8f | validation_loss=%.8f | best_validation=%.8f | patience=%d/%d | lr=%.6g%s",
            epoch + 1, total_epochs, train_loss, vl, best[0], stale,
            int(config["patience"]), opt.param_groups[0]["lr"],
            " | improved" if improved else "",
        )
        if stale>=config["patience"]:
            logging.info("NN early stopping at epoch %d; restoring best validation model.", epoch + 1)
            break
    model.load_state_dict(best[1]); reports={}; model.eval()
    with torch.no_grad():
        for name,rows in zip(("train","validation","test"),splits):
            x,y=ds(rows).tensors; x,y=x.to(device),y.to(device); pred=model(x); reports[name]={"loss":float(loss_fn(pred,y)),**metrics(y.cpu().numpy(),pred.cpu().numpy()),"samples":len(rows),"prediction_min":float(pred.min()),"prediction_max":float(pred.max())}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"model_config":{"vocab_size":len(enc.vocab),"embedding_dim":config["embedding_dim"],"hidden_dim":config["hidden_dim"]}},out/"model.pt"); (out/"encoder.json").write_text(json.dumps(enc.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8"); meta={"seed":seed,"device":str(device),"unique_structures":n,"target":"clipped LM-fitted R2","target_clip_range":[float(config.get("r2_clip_min",-10.0)),float(config.get("r2_clip_max",1.0))],"clipped_records":clipped_count,"metrics":reports,"history":history}; (out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); return meta


def load_expanded_predictor(model_dir):
    device=resolve_device(); root=Path(model_dir); enc=TreeEncoder.from_dict(json.loads((root/"encoder.json").read_text())); ck=torch.load(root/"model.pt",map_location=device,weights_only=True); model=BoundedStructureRegressor(**ck["model_config"]).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    def predict(formulas):
        with torch.no_grad(): return model(torch.tensor([enc.encode(f) for f in formulas],device=device)).cpu().numpy()
    return predict
