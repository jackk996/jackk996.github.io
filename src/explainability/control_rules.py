def risk_level(score):
    if not 0 <= score <= 1: raise ValueError('risk score must be in [0,1]')
    if score < 0.25: return '低风险'
    if score < 0.50: return '中风险'
    if score < 0.75: return '高风险'
    return '极高风险'
ACTIONS={'低风险':['常态监控','月度简报'],'中风险':['当日核实异常原因','制定措施','纳入风险登记册','每周跟踪'],'高风险':['成立专项组','暂停非关键任务','集中资源整改','每两天汇报'],'极高风险':['项目总监接管','暂停项目活动','专家诊断','重制计划并与客户协商']}
def recommended_actions(score): return ACTIONS[risk_level(score)]
