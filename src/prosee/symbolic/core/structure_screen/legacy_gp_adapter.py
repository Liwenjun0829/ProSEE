"""Legacy-compatible GP generation/output, without importing or changing legacy search."""
import json, logging
from pathlib import Path
import numpy as np
import sympy as sp

from .encoding import canonical_structure
from .tree_search_ops import delete_subtree_mutation, nodes_with_paths, replace_at_path
from .lm import fit_constants


def _depth(e): return 1+max((_depth(a) for a in e.args),default=0)
def _predict(e,X):
    try:
        symbols=sorted(e.free_symbols,key=lambda s:s.name); cols=[int(str(s)[1:])-1 for s in symbols]
        if any(i<0 or i>=X.shape[1] for i in cols): return None
        f=sp.lambdify(symbols,e,"numpy")
        with np.errstate(all="ignore"): v=np.asarray(f(*[X[:,i] for i in cols]),float)
        if v.ndim==0:v=np.full(len(X),float(v))
        v=v.reshape(-1)
        return v if v.shape==(len(X),) and np.all(np.isfinite(v)) and np.std(v)>1e-12 and np.max(np.abs(v))<1e150 else None
    except Exception:return None


def _valid(e,X,seen,max_complexity):
    try:
        e=sp.sympify(e); allowed={sp.Symbol(f"x{i+1}") for i in range(X.shape[1])}
        if not e.free_symbols or not e.free_symbols<=allowed or sp.count_ops(e)>max_complexity or _depth(e)>10:return None
        if any(n.func not in {sp.log,sp.sin,sp.cos,sp.exp,sp.Abs} for n in e.atoms(sp.Function)):return None
        if any(n.is_Pow and n.exp.is_number and abs(float(n.exp))>6 for n in sp.preorder_traversal(e)):return None
        key=canonical_structure(e)
        if key in seen or _predict(e,X) is None:return None
        seen.add(key);return e
    except Exception:return None


def _replace(e,replacement,rng,nonroot=False):
    nodes=nodes_with_paths(e,include_root=not nonroot)
    if not nodes:return e
    path,_=nodes[int(rng.integers(len(nodes)))]
    return replace_at_path(e,path,replacement)


def _subtree(rng,nvars):
    variables=[sp.Symbol(f"x{i+1}") for i in range(nvars)]; choice=int(rng.integers(7)); x=variables[int(rng.integers(nvars))]
    return x if choice<2 else sp.Float(rng.uniform(-3,3)) if choice==2 else (sp.log(sp.Abs(x)+1),sp.sin(x),sp.cos(x),sp.exp(-sp.Abs(x)))[choice-3]


def _mutate(e,rng,X,delete_probability,max_complexity):
    if rng.random()<delete_probability:
        # Adapter keeps the independent delete-subtree probability requested by the new flow.
        return delete_subtree_mutation(e,__import__("random").Random(int(rng.integers(2**31))),10,max_complexity*4,tuple(f"x{i+1}" for i in range(X.shape[1])))
    nodes=nodes_with_paths(e); mode=int(rng.integers(4))
    if mode==0:
        syms=[item for item in nodes if item[1].is_Symbol]
        if syms:
            path,_=syms[int(rng.integers(len(syms)))]
            return replace_at_path(e,path,sp.Symbol(f"x{int(rng.integers(X.shape[1]))+1}"))
    elif mode==1:
        nums=[item for item in nodes if item[1].is_Number]
        if nums:
            path,old=nums[int(rng.integers(len(nums)))]
            return replace_at_path(e,path,sp.Float(float(old)+rng.normal(0,.5)))
    elif mode==2:
        binary=[item for item in nodes if item[1].func in (sp.Add,sp.Mul) and len(item[1].args)>=2]
        if binary:
            path,old=binary[int(rng.integers(len(binary)))]
            replacement=(sp.Mul(*old.args,evaluate=False) if old.func==sp.Add else sp.Add(*old.args,evaluate=False))
            return replace_at_path(e,path,replacement)
    return _replace(e,_subtree(rng,X.shape[1]),rng)


def run_legacy_compatible_gp(base_formulas,predictor,X,y,output_path,config,lm_config,seed=42):
    rng=np.random.default_rng(seed); seen=set(); population=[]
    for text in base_formulas:
        e=_valid(text,X,seen,config["max_complexity"])
        if e is not None:population.append(e)
    if not population:raise RuntimeError("no valid GP seed formulas")
    population=population[:config["population_size"]]; generated={}
    for generation in range(config["generations"]):
        pool=population+list(generated.values()); predicted=np.asarray(predictor([str(e) for e in pool]),float); order=np.argsort(np.nan_to_num(predicted,nan=-np.inf))[::-1]; elites=[pool[i] for i in order[:config["elite_size"]]]; nxt=list(elites); attempts=0
        while len(nxt)<config["population_size"] and attempts<config["population_size"]*20:
            attempts+=1; parent=pool[int(rng.choice(order[:max(1,min(len(order),config["tournament_pool"]))]))]; child=parent
            if rng.random()<config["crossover_rate"]:
                donor=pool[int(rng.integers(len(pool)))]; donor_nodes=list(sp.preorder_traversal(donor)); child=_replace(child,donor_nodes[int(rng.integers(len(donor_nodes)))],rng)
            if rng.random()<config["mutation_rate"]:child=_mutate(child,rng,X,config["delete_subtree_probability"],config["max_complexity"])
            valid=_valid(child,X,seen,config["max_complexity"])
            if valid is not None:nxt.append(valid);generated[canonical_structure(valid)]=valid
        population=nxt;logging.info("GP generation=%d population=%d unique=%d",generation+1,len(population),len(generated))
    candidates=list(generated.values()); scores=np.asarray(predictor([str(e) for e in candidates]),float); ranked=sorted(zip(candidates,scores),key=lambda x:float(np.nan_to_num(x[1],nan=-np.inf)),reverse=True)[:config["screen_top_k"]]
    results=[]
    for rank,(expr,predicted) in enumerate(ranked,1):
        fitted=fit_constants(str(expr),X,y,lm_config["restarts"],lm_config["max_nfev"],seed+rank)
        if fitted:results.append({"rank":rank,"formula":fitted["optimized_formula"],"source_formula":str(expr),"structure":canonical_structure(expr),"predicted_r2":min(float(predicted),1.0),"fitted_r2":fitted["true_r2"],"optimized_constants":fitted["constants"],"complexity":int(sp.count_ops(expr)),"formula_node_count":sum(1 for _ in sp.preorder_traversal(expr)),"candidate_source":"ga_generated"})
    results.sort(key=lambda x:(-x["fitted_r2"],x["complexity"]));results=results[:config["top_k"]]
    for i,r in enumerate(results,1):r["rank"]=i
    payload={"metadata":{"seed":seed,"stage":"neural_prescreen_then_lm","generated_count":len(generated),"screened_count":len(ranked),"saved_count":len(results),"ranking":["fitted_r2_desc","complexity_asc"],"xgboost":False,"icl":False,"gp":config},"results":results}
    Path(output_path).parent.mkdir(parents=True,exist_ok=True);Path(output_path).write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8");return payload
