from __future__ import annotations
from dataclasses import dataclass

ID_COLUMNS = ["project_id", "subsystem_id", "month"]
TARGET_COLUMN = "risk_score"
INDICATOR_COLUMNS = [
    "milestone_delay_rate", "module_delay_days", "total_schedule_variance", "requirement_change_count",
    "code_defect_density", "escaped_defect_rate", "functional_compliance_rate", "stability_incident_count",
    "budget_overrun_rate", "labor_hour_deviation", "rework_cost_ratio", "resource_idle_rate",
    "key_staff_turnover_rate", "workload_saturation_deviation", "attendance_anomaly_rate", "skill_mismatch_rate",
    "technical_debt_density", "mean_fan_in_out", "dependency_risk_index", "tech_stack_deviation",
]
CHINESE_MAPPING = {
    "milestone_delay_rate": "里程碑延期率", "module_delay_days": "模块延期天数", "total_schedule_variance": "总进度偏差", "requirement_change_count": "需求变更次数",
    "code_defect_density": "代码缺陷密度", "escaped_defect_rate": "逃逸缺陷率", "functional_compliance_rate": "功能符合率", "stability_incident_count": "稳定性事件次数",
    "budget_overrun_rate": "预算超支率", "labor_hour_deviation": "工时偏差", "rework_cost_ratio": "返工成本占比", "resource_idle_rate": "资源闲置率",
    "key_staff_turnover_rate": "关键人员流失率", "workload_saturation_deviation": "工作负荷饱和度偏差", "attendance_anomaly_rate": "考勤异常率", "skill_mismatch_rate": "技能不匹配率",
    "technical_debt_density": "技术债密度", "mean_fan_in_out": "平均扇入扇出", "dependency_risk_index": "依赖风险指数", "tech_stack_deviation": "技术栈偏离度",
}
DIMENSIONS = {"进度": INDICATOR_COLUMNS[:4], "质量": INDICATOR_COLUMNS[4:8], "成本": INDICATOR_COLUMNS[8:12], "人力": INDICATOR_COLUMNS[12:16], "技术": INDICATOR_COLUMNS[16:]}
REQUIRED_COLUMNS = ID_COLUMNS + [TARGET_COLUMN] + INDICATOR_COLUMNS

@dataclass(frozen=True)
class IndicatorMeta:
    name: str
    dimension: str
    direction: str = "positive"
    legal_min: float | None = None
    legal_max: float | None = None
    missing_strategy: str = "median"

def default_indicator_metadata():
    metadata = {}
    for dim, cols in DIMENSIONS.items():
        for c in cols:
            direction = "inverse" if c == "functional_compliance_rate" else "bidirectional" if c == "workload_saturation_deviation" else "positive"
            metadata[c] = IndicatorMeta(c, dim, direction, None, None, "median")
    return metadata
