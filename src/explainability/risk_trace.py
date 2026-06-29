import numpy as np
from .control_rules import risk_level, recommended_actions

def explain_prediction(sample_id, metadata, pca_features, attention_weights, predicted_score, pca_loadings, indicator_names):
    row=metadata.iloc[sample_id].to_dict(); att=np.asarray(attention_weights, dtype=float); months=row['input_months']
    order=np.argsort(att)[::-1]; crit=[months[order[0]]]
    if len(order)>1 and att[order[0]]-att[order[1]] < 0.05: crit.append(months[order[1]])
    key_idx=order[0]; F=np.asarray(pca_features[sample_id][key_idx], dtype=float); contrib=np.abs(F)/max(np.abs(F).sum(),1e-12)
    dom=int(np.argmax(contrib)); loads=np.asarray(pca_loadings)[:,dom]; top=np.argsort(np.abs(loads))[::-1][:3]
    level=risk_level(float(predicted_score)); result={**row,'predicted_score':float(predicted_score),'risk_level':level,'critical_months':crit,'attention_weights':dict(zip(months,att.tolist())),'dominant_component':dom+1,'component_contributions':contrib.tolist(),'top3_indicators':[{'indicator':indicator_names[i],'loading':float(loads[i])} for i in top],'recommended_actions':recommended_actions(float(predicted_score))}
    md=f"# 风险溯源报告\n\n项目：{row['project_id']}，子系统：{row['subsystem_id']}，目标月份：{row['target_month']}。预测风险：{predicted_score:.4f}（{level}）。关键累积月份：{', '.join(crit)}。主导主成分：PC{dom+1}。Top-3指标：{', '.join(x['indicator'] for x in result['top3_indicators'])}。"
    return result, md
