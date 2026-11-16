import csv, json, logging, math, random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .encoding import TreeEncoder, canonical_structure
from .model import StructureRegressor


def resolve_device(config=None):
    """Use CUDA when available, with an automatic CPU fallback."""
    requested = str((config or {}).get("device", "auto")).lower()
    if requested == "cpu":
        return torch.device("cpu")
    if requested in ("auto", "cuda") and torch.cuda.is_available():
        return torch.device("cuda")
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError(f"unsupported device: {requested}")
    return torch.device("cpu")


def load_records(path: Path):
    if path.suffix.lower() == ".csv":
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    else:
        value = json.loads(path.read_text(encoding="utf-8")); rows = value if isinstance(value, list) else value.get("records", value.get("samples", []))
    out = {}
    for row in rows:
        formula = row.get("formula") or row.get("expr") or row.get("structure")
        score = row.get("best_r2", row.get("fitted_r2", row.get("true_r2")))
        try:
            score = float(score); key = canonical_structure(formula)
            if formula and math.isfinite(score) and key not in out: out[key] = (formula, score)
        except (TypeError, ValueError, SyntaxError): pass
    return list(out.values())


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p); e = y-p
    den = np.sum((y-y.mean())**2)
    return {"mae": float(np.mean(abs(e))), "rmse": float(np.sqrt(np.mean(e**2))), "r2": float(1-np.sum(e**2)/den) if den else float("nan")}


def train(input_path, output_dir, config):
    seed=int(config["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device=resolve_device(config); logging.info("NN training device=%s",device)
    records=load_records(Path(input_path))
    if len(records)<6: raise ValueError("at least 6 unique valid formula structures are required")
    rng=np.random.default_rng(seed); rng.shuffle(records); n=len(records); a=int(n*config["split"][0]); b=a+int(n*config["split"][1]); splits=[records[:a],records[a:b],records[b:]]
    enc=TreeEncoder(config["max_nodes"]).fit([x[0] for x in splits[0]])
    def ds(items): return TensorDataset(torch.tensor([enc.encode(x[0]) for x in items]),torch.tensor([x[1] for x in items],dtype=torch.float32))
    model=StructureRegressor(len(enc.vocab),config["embedding_dim"],config["hidden_dim"]).to(device); opt=torch.optim.Adam(model.parameters(),lr=config["learning_rate"]); loss_fn=torch.nn.MSELoss(); best=None; stale=0; history=[]
    for epoch in range(config["epochs"]):
        model.train(); losses=[]
        for x,y in DataLoader(ds(splits[0]),batch_size=config["batch_size"],shuffle=True): x,y=x.to(device),y.to(device); opt.zero_grad(); loss=loss_fn(model(x),y); loss.backward(); opt.step(); losses.append(loss.item())
        model.eval()
        with torch.no_grad(): vx,vy=ds(splits[1]).tensors; vx,vy=vx.to(device),vy.to(device); vl=loss_fn(model(vx),vy).item()
        history.append({"epoch":epoch+1,"train_loss":float(np.mean(losses)),"validation_loss":vl}); logging.info("epoch=%d train_loss=%.6g validation_loss=%.6g",epoch+1,history[-1]["train_loss"],vl)
        if best is None or vl<best[0]: best=(vl,{k:v.detach().clone() for k,v in model.state_dict().items()}); stale=0
        else: stale+=1
        if stale>=config["patience"]: break
    model.load_state_dict(best[1]); reports={}
    model.eval()
    with torch.no_grad():
        for name,items in zip(("train","validation","test"),splits): x,y=ds(items).tensors; x,y=x.to(device),y.to(device); pred=model(x); reports[name]={"loss":float(loss_fn(pred,y)),**metrics(y.cpu().numpy(),pred.cpu().numpy()),"samples":len(items)}
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True); torch.save({"state_dict":{k:v.detach().cpu() for k,v in model.state_dict().items()},"model_config":{"vocab_size":len(enc.vocab),"embedding_dim":config["embedding_dim"],"hidden_dim":config["hidden_dim"]}},out/"model.pt"); (out/"encoder.json").write_text(json.dumps(enc.to_dict(),ensure_ascii=False,indent=2),encoding="utf-8"); meta={"seed":seed,"device":str(device),"unique_structures":n,"metrics":reports,"history":history,"target":"R2 after full constant fitting"}; (out/"metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8"); return meta


def load_predictor(model_dir):
    device=resolve_device(); root=Path(model_dir); enc=TreeEncoder.from_dict(json.loads((root/"encoder.json").read_text())); ck=torch.load(root/"model.pt",map_location=device,weights_only=True); model=StructureRegressor(**ck["model_config"]).to(device); model.load_state_dict(ck["state_dict"]); model.eval()
    def predict(formulas):
        with torch.no_grad(): return model(torch.tensor([enc.encode(f) for f in formulas],device=device)).cpu().numpy()
    return predict
