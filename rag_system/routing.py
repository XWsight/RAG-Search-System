"""Query-intent and evidence routing kept independent from retrieval engines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from rag_system.config import Settings
from rag_system.domain import Route, RouteDecision, SearchHit


class QueryIntent(StrEnum):
    """Capability requirement detected before evidence confidence is applied."""

    KNOWLEDGE = "knowledge"
    FRESHNESS = "freshness"
    UNSUPPORTED_ACTION = "unsupported_action"
    RESTRICTED = "restricted"


@dataclass(frozen=True, slots=True)
class QueryIntentAssessment:
    intent: QueryIntent
    rule_id: str


class QueryIntentClassifier(Protocol):
    def classify(self, question: str) -> QueryIntentAssessment: ...


class RuleBasedQueryIntentClassifier:
    """Conservative capability classifier with stable, auditable rule IDs."""

    _EXPLICIT_EGRESS = (
        "联网",
        "网络搜索",
        "搜索网络",
        "从网络",
        "从官网",
        "实时来源",
        "允许使用网络",
        "online search",
        "search the web",
        "from the web",
    )
    _FRESHNESS_TERMS = (
        "实时",
        "今天",
        "明天",
        "今晚",
        "本周",
        "最近七天",
        "最近",
        "当前",
        "最新",
        "此刻",
        "过去二十四小时",
        "未来",
        "未来三天",
        "当前最新",
        "最新稳定",
        "最新正式",
        "最近一次发布",
        "right now",
        "today",
        "tomorrow",
        "latest release",
    )
    _DYNAMIC_TOPICS = (
        "天气",
        "温度",
        "下雨",
        "空气质量",
        "汇率",
        "报价",
        "价格",
        "堵车",
        "延误",
        "版本",
        "发布",
        "新闻",
        "动态",
        "余额",
        "进度",
        "weather",
        "exchange rate",
        "traffic",
        "release",
        "news",
    )
    _PRIVATE_RESOURCES = (
        "银行卡",
        "校园卡",
        "私人账户",
        "个人账户",
        "医保账户",
        "教务系统",
        "学校网站",
        "身份证",
        "邮箱",
        "网盘",
        "云盘",
        "个人课表",
        "我的账户",
        "bank account",
        "email account",
        "private account",
    )
    _POLICY_DISCUSSION = (
        "allow_web",
        "allow_cloud",
        "外部调用开关",
        "联网搜索许可",
        "开关分别授权",
        "外发",
        "本地检索默认",
        "默认会不会",
        "默认值和作用范围",
        "系统可以从拒答",
        "egress policy",
        "privacy switch",
    )
    _RESOURCE_ACTIONS = (
        "读取",
        "查询",
        "查一下",
        "找出",
        "登录",
        "下载",
        "告诉",
        "删除",
        "read",
        "query",
        "log in",
        "download",
        "delete",
    )
    _DIRECT_ACTIONS = (
        ("发送", "邮件"),
        ("发送", "短信"),
        ("预订", "机票"),
        ("预订", "酒店"),
        ("购买", "机票"),
        ("转账",),
        ("付款",),
        ("支付",),
        ("send", "email"),
        ("book", "flight"),
        ("transfer", "money"),
    )
    _SECRET_EXTRACTION = (
        ("环境变量", "api key"),
        ("其他租户", "访问密钥"),
        ("隐藏配置", "数据库密码"),
        ("忽略权限", "认证令牌"),
        ("another tenant", "api key"),
        ("environment variable", "secret"),
        ("database password", "show"),
    )
    _MEDICAL_SUBJECTS = (
        "胸痛",
        "皮疹",
        "头晕",
        "处方药",
        "药剂量",
        "症状",
        "chest pain",
        "prescription dose",
        "diagnose my",
    )
    _MEDICAL_DIRECTIVES = (
        "诊断",
        "直接判断",
        "确定一种",
        "一定是",
        "不用医生",
        "不要查医学资料",
        "diagnose",
        "prescribe",
        "without a doctor",
    )

    def classify(self, question: str) -> QueryIntentAssessment:
        normalized = " ".join((question or "").casefold().split())
        if not normalized:
            return QueryIntentAssessment(QueryIntent.KNOWLEDGE, "empty_or_unspecified")
        if any(all(term in normalized for term in group) for group in self._DIRECT_ACTIONS):
            return QueryIntentAssessment(
                QueryIntent.UNSUPPORTED_ACTION,
                "direct_external_action",
            )
        if any(all(term in normalized for term in group) for group in self._SECRET_EXTRACTION):
            return QueryIntentAssessment(QueryIntent.RESTRICTED, "secret_extraction")
        if self._contains_any(normalized, self._MEDICAL_SUBJECTS) and self._contains_any(
            normalized, self._MEDICAL_DIRECTIVES
        ):
            return QueryIntentAssessment(QueryIntent.RESTRICTED, "medical_diagnosis")
        if self._contains_any(normalized, self._PRIVATE_RESOURCES) and self._contains_any(
            normalized, self._RESOURCE_ACTIONS
        ):
            return QueryIntentAssessment(
                QueryIntent.UNSUPPORTED_ACTION,
                "private_resource_action",
            )
        if self._contains_any(normalized, self._POLICY_DISCUSSION):
            return QueryIntentAssessment(QueryIntent.KNOWLEDGE, "capability_policy_question")
        if self._contains_any(normalized, self._EXPLICIT_EGRESS):
            return QueryIntentAssessment(QueryIntent.FRESHNESS, "explicit_egress_request")
        if self._contains_any(normalized, self._FRESHNESS_TERMS) and self._contains_any(
            normalized, self._DYNAMIC_TOPICS
        ):
            return QueryIntentAssessment(QueryIntent.FRESHNESS, "dynamic_information_request")
        return QueryIntentAssessment(QueryIntent.KNOWLEDGE, "knowledge_evidence")

    @staticmethod
    def _contains_any(value: str, terms: Sequence[str]) -> bool:
        return any(term in value for term in terms)


@dataclass(frozen=True, slots=True)
class RoutingSignal:
    """Bounded, content-free evidence used to explain one routing decision."""

    top_score: float
    second_score: float
    margin: float
    ranker_agreement: bool
    lexical_score: float
    lexical_support: float
    confidence: float
    query_intent: QueryIntent
    intent_rule: str

    @classmethod
    def empty(cls, intent: QueryIntentAssessment | None = None) -> RoutingSignal:
        assessment = intent or QueryIntentAssessment(
            QueryIntent.KNOWLEDGE,
            "empty_or_unspecified",
        )
        return cls(
            0.0,
            0.0,
            0.0,
            False,
            0.0,
            0.0,
            0.0,
            assessment.intent,
            assessment.rule_id,
        )

    def to_dict(self) -> dict[str, float | bool | str]:
        return {
            "top_score": round(self.top_score, 12),
            "second_score": round(self.second_score, 12),
            "margin": round(self.margin, 12),
            "ranker_agreement": self.ranker_agreement,
            "lexical_score": round(self.lexical_score, 12),
            "lexical_support": round(self.lexical_support, 12),
            "confidence": round(self.confidence, 12),
            "query_intent": self.query_intent.value,
            "intent_rule": self.intent_rule,
        }


@dataclass(frozen=True, slots=True)
class RoutingAssessment:
    """Public route decision paired with privacy-safe diagnostic evidence."""

    decision: RouteDecision
    signal: RoutingSignal


class RoutingPolicy:
    """Resolve capability intent before applying calibrated evidence confidence."""

    def __init__(
        self,
        settings: Settings,
        *,
        intent_classifier: QueryIntentClassifier | None = None,
    ) -> None:
        self.settings = settings.validate()
        self.intent_classifier = intent_classifier or RuleBasedQueryIntentClassifier()

    def decide(
        self,
        hits: Sequence[SearchHit],
        *,
        allow_web: bool,
        question: str = "",
    ) -> RouteDecision:
        return self.assess(hits, allow_web=allow_web, question=question).decision

    def assess(
        self,
        hits: Sequence[SearchHit],
        *,
        allow_web: bool,
        question: str = "",
    ) -> RoutingAssessment:
        intent = self.intent_classifier.classify(question)
        signal = self.signal(hits, intent=intent)
        confidence = signal.confidence

        if intent.intent is QueryIntent.RESTRICTED:
            return RoutingAssessment(
                RouteDecision(
                    Route.REFUSED,
                    confidence,
                    "请求超出当前系统允许的安全能力边界。",
                ),
                signal,
            )
        if intent.intent is QueryIntent.UNSUPPORTED_ACTION:
            return RoutingAssessment(
                RouteDecision(
                    Route.REFUSED,
                    confidence,
                    "请求需要当前系统未授权的外部账户或执行能力。",
                ),
                signal,
            )
        if intent.intent is QueryIntent.FRESHNESS:
            return RoutingAssessment(
                RouteDecision(
                    Route.WEB if allow_web else Route.REFUSED,
                    confidence,
                    "问题需要实时外部信息。" if allow_web else "问题需要实时外部信息但未授权联网。",
                ),
                signal,
            )
        if not hits:
            return RoutingAssessment(
                RouteDecision(
                    route=Route.WEB if allow_web else Route.REFUSED,
                    confidence=0.0,
                    reason="本地检索没有返回候选证据。",
                ),
                signal,
            )
        if confidence >= self.settings.local_confidence_threshold:
            return RoutingAssessment(
                RouteDecision(Route.LOCAL, confidence, "本地证据达到置信度阈值。"),
                signal,
            )
        hybrid_threshold = (
            self.settings.local_confidence_threshold * self.settings.hybrid_confidence_ratio
        )
        if allow_web and confidence >= hybrid_threshold:
            return RoutingAssessment(
                RouteDecision(Route.HYBRID, confidence, "本地证据不完整，将补充网络来源。"),
                signal,
            )
        if allow_web:
            return RoutingAssessment(
                RouteDecision(Route.WEB, confidence, "本地证据不足，将使用网络来源。"),
                signal,
            )
        return RoutingAssessment(
            RouteDecision(Route.REFUSED, confidence, "本地证据不足且联网搜索未开启。"),
            signal,
        )

    def confidence(self, hits: Sequence[SearchHit]) -> float:
        return self.signal(hits).confidence

    def signal(
        self,
        hits: Sequence[SearchHit],
        *,
        intent: QueryIntentAssessment | None = None,
    ) -> RoutingSignal:
        assessment = intent or QueryIntentAssessment(
            QueryIntent.KNOWLEDGE,
            "knowledge_evidence",
        )
        if not hits:
            return RoutingSignal.empty(assessment)
        top_score = max(0.0, min(1.0, hits[0].score))
        second_score = max(0.0, min(1.0, hits[1].score)) if len(hits) > 1 else 0.0
        margin = max(0.0, top_score - second_score)
        ranker_agreement = {"dense", "sparse"}.issubset(hits[0].reasons)
        lexical_score = max(0.0, min(1.0, hits[0].lexical_score or 0.0))
        lexical_support = min(
            1.0,
            lexical_score / self.settings.routing_lexical_saturation,
        )
        supported_agreement = float(ranker_agreement) * lexical_support
        confidence = min(
            1.0,
            0.75 * top_score
            + 0.15 * supported_agreement
            + 0.10 * min(1.0, margin * 4),
        )
        return RoutingSignal(
            top_score=top_score,
            second_score=second_score,
            margin=margin,
            ranker_agreement=ranker_agreement,
            lexical_score=lexical_score,
            lexical_support=lexical_support,
            confidence=float(confidence),
            query_intent=assessment.intent,
            intent_rule=assessment.rule_id,
        )


__all__ = [
    "QueryIntent",
    "QueryIntentAssessment",
    "QueryIntentClassifier",
    "RoutingAssessment",
    "RoutingPolicy",
    "RoutingSignal",
    "RuleBasedQueryIntentClassifier",
]
