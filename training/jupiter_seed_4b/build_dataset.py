#!/usr/bin/env python3
"""
Jupiter Seed 4B — Gold Dataset Builder v4
==========================================
FUNDAMENTAL REBUILD — Version 2.0.0

Principles:
  1. No artificial uniqueness markers in any natural-language field
  2. Every example is substantively distinct (different required answer)
  3. Content families are assigned before splitting
  4. Benchmark is built independently from training
  5. Deduplication runs on canonicalized text
  6. Honest dataset size: reports actual distinct count, does not fabricate

Each example has a scenario_brief describing:
  - Decision or task
  - Core facts
  - Required reasoning or transformation
  - Expected knowledge
  - Risk or ambiguity
  - Why the example is distinct from others

No two examples share the same scenario brief after normalization.

Language targets (aspirational, subject to honest count):
  Train: ar=270, en=180, ar-en=150
  Bench: ar=113, en=75,  ar-en=62

Domain targets (aspirational):
  IF=170, GB=128, TC=128, GR=128, EL=85, ED=128, AC=83

No teacher outputs. No paid APIs. No model downloads. No GPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import canonicalization engine
_TRAINING_DIR = Path(__file__).parent
sys.path.insert(0, str(_TRAINING_DIR))
from canonicalize import (
    canonicalize,
    canonical_hash,
    has_artificial_marker,
    run_deduplication,
    sha256_of,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DATASET_VERSION = "2.0.0"
SCHEMA_VERSION = "4.0"


def ar_char_count(text: str) -> int:
    return len(re.findall(r"[\u0600-\u06FF]", str(text)))


def en_char_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z]", str(text)))


def check_language_contract(lang: str, prompt: str, response: str) -> bool:
    ar_resp = ar_char_count(response)
    en_resp = en_char_count(response)
    combined_ar = ar_char_count(prompt + " " + response)
    combined_en = en_char_count(prompt + " " + response)
    if lang == "ar":
        return ar_resp >= 30 and (ar_resp / max(len(response.replace(" ", "")), 1)) >= 0.20
    elif lang == "en":
        return en_resp >= 30
    elif lang == "ar-en":
        return combined_ar >= 20 and combined_en >= 20
    return True


# ---------------------------------------------------------------------------
# EXAMPLE LIBRARY — each entry is a unique, substantively distinct example
#
# Format: dict with all required fields
# No artificial markers. No variant tags. No reference suffixes.
# ---------------------------------------------------------------------------

def make_example(
    family: str,
    lang: str,
    domain: str,
    task_type: str,
    difficulty: str,
    country: str,
    prov_type: str,
    source_name: str,
    source_url: str,
    scenario_brief: str,
    prompt: str,
    response: str,
    factual_dep: bool = False,
) -> Dict[str, Any]:
    """Create a single example dict. Validates no artificial markers."""
    assert not has_artificial_marker(prompt), f"Artificial marker in prompt for family {family}"
    assert not has_artificial_marker(response), f"Artificial marker in response for family {family}"
    assert scenario_brief.strip(), f"Empty scenario_brief for family {family}"
    return {
        "content_family_id": family,
        "language": lang,
        "domain": domain,
        "task_type": task_type,
        "difficulty": difficulty,
        "country_or_region": country,
        "provenance_type": prov_type,
        "source_name": source_name,
        "source_url": source_url,
        "source_access_date": "2024-01-01",
        "factual_dependency": factual_dep,
        "scenario_brief": scenario_brief,
        "prompt": prompt,
        "response": response,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING LIBRARY — examples assigned to TRAIN split
# ═══════════════════════════════════════════════════════════════════════════

TRAIN_EXAMPLES: List[Dict] = []

# ─── Islamic Finance — Training ───────────────────────────────────────────

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-001", lang="ar", domain="islamic_finance",
    task_type="islamic_finance_terminology", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Define Murabaha. "
        "Core facts: AAOIFI standard definition. "
        "Required reasoning: Recall the key structural elements of a Murabaha contract. "
        "Expected knowledge: Cost disclosure, profit margin, bank ownership requirement. "
        "Risk: Confusing Murabaha with a conventional loan. "
        "Distinct from: All other IF examples because it asks for the foundational definition."
    ),
    prompt="ما تعريف عقد المرابحة وفق معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية؟",
    response=(
        "المرابحة وفق معايير أيوفي عقد بيع يُفصح فيه البائع عن تكلفة السلعة وهامش الربح المتفق عليه. "
        "يشترط المعيار أن يكون البائع مالكاً فعلياً للسلعة قبل البيع، وأن يكون الثمن الآجل محدداً ومعلوماً. "
        "تُعدّ المرابحة من أكثر صيغ التمويل الإسلامي استخداماً في البنوك الخليجية لوضوح هيكلها القانوني."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-002", lang="ar", domain="islamic_finance",
    task_type="islamic_finance_terminology", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Define Musharaka. "
        "Core facts: AAOIFI standard definition of partnership finance. "
        "Required reasoning: Distinguish profit-sharing ratio from loss-sharing ratio. "
        "Expected knowledge: Capital contribution, profit/loss allocation rules. "
        "Risk: Confusing Musharaka with Mudaraba. "
        "Distinct from IF-TRAIN-001: Asks about a different Islamic finance contract."
    ),
    prompt="ما تعريف عقد المشاركة في التمويل الإسلامي وكيف تُوزَّع الأرباح والخسائر؟",
    response=(
        "المشاركة عقد شراكة بين طرفين أو أكثر يُقدّم كل منهم رأس مال، ويُوزَّع الربح وفق نسبة متفق عليها. "
        "أما الخسارة فتُوزَّع بنسبة رأس المال المُقدَّم من كل شريك دون استثناء. "
        "يختلف هذا العقد عن المضاربة في أن جميع الشركاء يُقدّمون رأس المال، "
        "بينما في المضاربة يُقدّم رب المال رأس المال وحده ويُقدّم المضارب العمل."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-003", lang="ar", domain="islamic_finance",
    task_type="islamic_finance_terminology", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Define Mudaraba. "
        "Core facts: AAOIFI definition of profit-sharing with one capital provider. "
        "Required reasoning: Explain who bears financial loss and why. "
        "Expected knowledge: Rabb al-mal vs mudarib roles, loss allocation. "
        "Risk: Confusing Mudaraba with Musharaka. "
        "Distinct from IF-TRAIN-001/002: Different contract with different loss allocation."
    ),
    prompt="ما تعريف عقد المضاربة في الفقه الإسلامي وما الفرق بين رب المال والمضارب؟",
    response=(
        "المضاربة عقد شراكة يُقدّم فيه رب المال رأس المال كاملاً، ويُقدّم المضارب الخبرة والعمل. "
        "يُوزَّع الربح وفق نسبة متفق عليها بين الطرفين، أما الخسارة المالية فتقع على رب المال وحده "
        "ما لم يثبت تقصير المضارب أو تعدّيه. "
        "يتحمل المضارب خسارة جهده ووقته في حال الخسارة دون تعويض مالي."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-004", lang="ar", domain="islamic_finance",
    task_type="islamic_finance_terminology", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Define Ijarah and explain its key Sharia requirements. "
        "Core facts: Lease contract where bank retains ownership. "
        "Required reasoning: Identify the ownership and maintenance obligations. "
        "Expected knowledge: Lessor's ownership, maintenance responsibility, prohibited conditions. "
        "Risk: Confusing Ijarah with a conventional lease that transfers all risk. "
        "Distinct from others: Focuses on lease-specific Sharia requirements."
    ),
    prompt="ما عقد الإجارة في التمويل الإسلامي وما المتطلبات الشرعية الرئيسية لصحته؟",
    response=(
        "الإجارة عقد تأجير يُخوّل المستأجر الانتفاع بأصل مملوك للمؤجر مقابل أجرة محددة. "
        "تشترط الشريعة الإسلامية لصحة عقد الإجارة: "
        "أولاً: أن يكون الأصل المؤجَّر مملوكاً للمؤجر فعلاً وقت التأجير. "
        "ثانياً: أن تكون الأجرة محددة ومعلومة لكلا الطرفين. "
        "ثالثاً: أن يتحمل المؤجر الصيانة الأساسية للأصل لأنه مالكه. "
        "رابعاً: أن يكون الأصل قابلاً للانتفاع دون استهلاكه."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-005", lang="ar", domain="islamic_finance",
    task_type="islamic_finance_terminology", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Define Istisna and explain its use in project finance. "
        "Core facts: Manufacturing contract for future delivery. "
        "Required reasoning: Explain how deferred payment is permissible. "
        "Expected knowledge: Specification requirements, price certainty, delivery obligations. "
        "Risk: Confusing Istisna with Salam (another forward contract). "
        "Distinct from others: Focuses on manufacturing/construction finance."
    ),
    prompt="ما عقد الاستصناع وكيف يُستخدم في تمويل مشاريع البنية التحتية؟",
    response=(
        "الاستصناع عقد تصنيع يُلتزم فيه الصانع بتصنيع سلعة أو إنشاء مشروع وفق مواصفات محددة مقابل ثمن متفق عليه. "
        "يُجيز الفقه الإسلامي تأجيل الثمن أو تقسيطه في الاستصناع خلافاً للقاعدة العامة. "
        "يُستخدم في تمويل مشاريع البنية التحتية من خلال هيكل الاستصناع الموازي: "
        "تُبرم المؤسسة المالية عقد استصناع مع المطوّر ثم تُبرم عقداً موازياً مع المقاول، "
        "مما يُتيح تمويل المشاريع الكبرى دون مخالفة أحكام الشريعة."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-006", lang="ar", domain="islamic_finance",
    task_type="risk_identification", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify Sharia risks in a hypothetical Ijarah Muntahia Bittamleek structure. "
        "Core facts: Fictional bank offering lease-to-own for commercial real estate. "
        "Required reasoning: Identify the four main Sharia risk categories specific to this structure. "
        "Expected knowledge: Ownership risk, contract independence, maintenance, transfer mechanism. "
        "Risk: Missing the requirement that the sale and lease contracts must be independent. "
        "Distinct from IF-TRAIN-004: Focuses on risks in a specific lease-to-own variant, not the basic definition."
    ),
    prompt=(
        "بنك إسلامي افتراضي يُقدّم منتج إجارة منتهية بالتمليك لتمويل عقارات تجارية. "
        "ما المخاطر الشرعية الرئيسية التي يجب على الهيئة الشرعية مراجعتها في هذا الهيكل؟"
    ),
    response=(
        "تشمل المخاطر الشرعية الرئيسية في الإجارة المنتهية بالتمليك لعقارات تجارية افتراضية: "
        "أولاً: مخاطر الملكية — يجب أن تظل المؤسسة المالية مالكةً للعقار طوال مدة الإجارة، "
        "وأي تصرف يُضعف هذه الملكية يُفسد العقد شرعاً. "
        "ثانياً: استقلالية العقدين — يجب أن يكون عقد الإجارة وعقد نقل الملكية مستقلَّين تماماً، "
        "ولا يجوز اشتراط أحدهما في الآخر. "
        "ثالثاً: الصيانة الأساسية — تقع على المؤجر ولا يجوز تحميلها للمستأجر في العقد الأصلي. "
        "رابعاً: آلية نقل الملكية — يجب أن تتم بعقد هبة أو بيع مستقل بعد انتهاء مدة الإجارة. "
        "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-007", lang="en", domain="islamic_finance",
    task_type="regulatory_explanation", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="IFSB", source_url="https://www.ifsb.org/",
    scenario_brief=(
        "Task: Explain IFSB capital adequacy standards for Islamic banks. "
        "Core facts: IFSB-2 adapts Basel III for Islamic finance risk profiles. "
        "Required reasoning: Explain why Murabaha and Musharaka have different risk weights. "
        "Expected knowledge: Asset ownership in Murabaha, equity risk in Musharaka. "
        "Risk: Applying conventional Basel III weights directly without Islamic adjustments. "
        "Distinct from others: Focuses on the specific Basel III adaptation for Islamic products."
    ),
    prompt="What capital adequacy standards apply to Islamic banks under the IFSB framework, and how do they differ from conventional Basel III?",
    response=(
        "The Islamic Financial Services Board (IFSB) has issued IFSB-2, the Capital Adequacy Standard "
        "for Institutions Offering Islamic Financial Services, which adapts Basel III concepts to "
        "account for the unique risk profiles of Islamic financing modes. "
        "For Murabaha, the credit risk treatment differs from conventional loans because the bank "
        "holds the asset before sale, creating commodity risk in addition to credit risk. "
        "For Musharaka and Mudaraba, equity risk weights apply because the bank participates in "
        "business risk rather than lending at a fixed rate. "
        "For Ijarah, the bank retains ownership risk throughout the lease period. "
        "These distinctions mean Islamic banks may face higher capital requirements for certain "
        "products than conventional equivalents. Specific current requirements should be verified "
        "against the latest IFSB published standards and applicable central bank regulations."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-008", lang="en", domain="islamic_finance",
    task_type="decision_comparison", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Compare Diminishing Musharaka and Ijarah Muntahia Bittamleek for home finance. "
        "Core facts: Two common GCC home finance structures with different ownership mechanics. "
        "Required reasoning: Identify the key structural difference in how ownership transfers. "
        "Expected knowledge: Progressive equity acquisition vs end-of-term transfer. "
        "Risk: Treating both as equivalent because both result in eventual ownership. "
        "Distinct from IF-TRAIN-006: Compares two structures rather than identifying risks in one."
    ),
    prompt="Compare Diminishing Musharaka and Ijarah Muntahia Bittamleek as home financing structures, focusing on how ownership transfers to the client.",
    response=(
        "In Diminishing Musharaka, the bank and client co-own the property from the outset. "
        "The client pays rent on the bank's share and simultaneously purchases additional equity "
        "units from the bank over time. Ownership transfers progressively as the client acquires "
        "each unit, so the client builds equity throughout the financing period. "
        "In Ijarah Muntahia Bittamleek, the bank retains full ownership throughout the lease term. "
        "The client pays rent only, with no equity accumulation during the lease. "
        "Ownership transfers at the end of the term via a separate gift or sale contract. "
        "The key structural difference: Diminishing Musharaka creates progressive co-ownership; "
        "Ijarah Muntahia Bittamleek creates a lease relationship with a deferred ownership event. "
        "Both require Sharia board approval and independent legal documentation of each stage. "
        "Note: This comparison uses fictional structures for training purposes."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-009", lang="ar-en", domain="islamic_finance",
    task_type="arabic_english_translation", difficulty="basic",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'Murabaha' and provide a bilingual definition. "
        "Core facts: The term is used in both Arabic and English in GCC banking. "
        "Required reasoning: Provide the Arabic term and a concise bilingual definition. "
        "Expected knowledge: The structural elements of a Murabaha contract. "
        "Risk: Providing an incomplete definition that omits the disclosure requirement. "
        "Distinct from IF-TRAIN-001: Bilingual translation task, not a monolingual definition."
    ),
    prompt="Translate the Islamic finance term 'Murabaha' into Arabic and provide a brief bilingual definition.",
    response=(
        "الترجمة: مرابحة.\n\n"
        "التعريف بالعربية: المرابحة عقد بيع يُفصح فيه البائع عن تكلفة السلعة وهامش الربح المتفق عليه، "
        "ويشترط أن يكون البائع مالكاً للسلعة قبل البيع.\n\n"
        "English definition: Murabaha is a cost-plus-profit sale contract in which the seller "
        "discloses the cost of the asset and the agreed profit margin to the buyer. "
        "The seller must own the asset before the sale is concluded."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-010", lang="ar-en", domain="islamic_finance",
    task_type="arabic_english_translation", difficulty="basic",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'Sukuk' and provide a bilingual definition. "
        "Core facts: Sukuk are Islamic investment certificates, not bonds. "
        "Required reasoning: Explain the asset-backing requirement that distinguishes Sukuk from bonds. "
        "Expected knowledge: Asset-backed structure, profit distribution vs interest. "
        "Risk: Describing Sukuk as simply 'Islamic bonds' without the asset-backing distinction. "
        "Distinct from IF-TRAIN-009: Different term with a structurally different definition."
    ),
    prompt="Translate the Islamic finance term 'Sukuk' into Arabic and explain the key difference from conventional bonds.",
    response=(
        "الترجمة: صكوك.\n\n"
        "التعريف بالعربية: الصكوك أدوات استثمارية إسلامية تمثّل حصصاً في ملكية أصول أو مشاريع حقيقية، "
        "ويحصل حاملها على عائد مرتبط بأداء هذه الأصول لا على فائدة ثابتة.\n\n"
        "Key difference from conventional bonds: Sukuk represent ownership in underlying assets "
        "or usufruct rights, whereas conventional bonds represent debt obligations. "
        "Sukuk returns are derived from asset performance or rental income, not predetermined interest. "
        "This asset-backing requirement is the fundamental Sharia distinction."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-011", lang="ar-en", domain="islamic_finance",
    task_type="arabic_english_translation", difficulty="intermediate",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'Takaful' and explain how it differs from conventional insurance. "
        "Core facts: Takaful is based on mutual donation and cooperation. "
        "Required reasoning: Identify the three prohibited elements in conventional insurance. "
        "Expected knowledge: Riba, gharar, maysir prohibition; tabarru concept. "
        "Risk: Describing Takaful as merely 'Islamic insurance' without the structural differences. "
        "Distinct from IF-TRAIN-009/010: Different term requiring explanation of three prohibited elements."
    ),
    prompt="Translate 'Takaful' into Arabic and explain the three elements of conventional insurance that it avoids.",
    response=(
        "الترجمة: تكافل.\n\n"
        "التعريف بالعربية: التكافل نظام تأمين إسلامي يقوم على مبدأ التبرع والتعاون المتبادل بين المشتركين، "
        "إذ يُودع كل مشترك جزءاً من قسطه في صندوق مشترك لتغطية الأضرار.\n\n"
        "The three prohibited elements avoided by Takaful:\n"
        "1. Riba (interest): Takaful funds are invested in Sharia-compliant assets only.\n"
        "2. Gharar (excessive uncertainty): The tabarru (donation) structure replaces the "
        "uncertain exchange of conventional insurance.\n"
        "3. Maysir (gambling): Participants contribute to a mutual fund rather than wagering "
        "on uncertain events for profit."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-012", lang="en", domain="islamic_finance",
    task_type="hallucination_resistance", difficulty="basic",
    country="GCC", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to provide a current profit rate for a specific bank. "
        "Core facts: Real-time financial data is not available; rates change frequently. "
        "Required reasoning: Explain why providing an invented rate is harmful. "
        "Expected knowledge: The risk of misleading financial decisions with invented data. "
        "Risk: Inventing a plausible-sounding rate that misleads the user. "
        "Distinct from others: Safety/refusal task requiring explicit refusal with explanation."
    ),
    prompt="What is the current profit rate offered by Al Rajhi Bank on Murabaha deposits?",
    response=(
        "I do not have access to real-time financial data and cannot provide current profit rates "
        "offered by any specific bank. Rates change frequently based on market conditions and "
        "central bank policy. "
        "Providing an invented rate would be harmful because it could mislead investment or "
        "savings decisions. "
        "For accurate current information, please visit Al Rajhi Bank's official website "
        "or contact their customer service directly."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-013", lang="ar", domain="islamic_finance",
    task_type="hallucination_resistance", difficulty="basic",
    country="GCC", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to confirm Sharia compliance of an unspecified product. "
        "Core facts: Sharia compliance depends on contract details not provided. "
        "Required reasoning: Explain what information is needed before any assessment. "
        "Expected knowledge: Compliance depends on structure, not just the label. "
        "Risk: Confirming compliance without reviewing the contract. "
        "Distinct from IF-TRAIN-012: Arabic refusal task about compliance, not profit rates."
    ),
    prompt="هل منتج التمويل الذي تُقدّمه شركة XYZ متوافق مع الشريعة الإسلامية؟",
    response=(
        "لا يمكنني تحديد مدى توافق أي منتج مع الشريعة الإسلامية دون الاطلاع على شروطه الكاملة وهيكله القانوني. "
        "التوافق الشرعي يعتمد على التفاصيل التعاقدية والأصول الأساسية والتفسير الفقهي المعتمد. "
        "لتقييم دقيق، يُرجى طلب الشهادة الشرعية والفتوى مباشرةً من الشركة المُصدِرة، "
        "أو استشارة هيئة رقابة شرعية مستقلة ومعتمدة."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-014", lang="ar", domain="islamic_finance",
    task_type="document_summarization", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Summarize the key principles of Istisna under AAOIFI standards. "
        "Core facts: AAOIFI Sharia Standard on Istisna. "
        "Required reasoning: Identify the three core principles: specification, price, delivery. "
        "Expected knowledge: Deferred payment permissibility, specification requirements. "
        "Risk: Confusing Istisna with Salam (which requires advance payment). "
        "Distinct from IF-TRAIN-005: Summary task vs definitional task; focuses on AAOIFI principles."
    ),
    prompt="لخّص المبادئ الأساسية لعقد الاستصناع وفق معايير هيئة أيوفي.",
    response=(
        "يُحدّد معيار أيوفي للاستصناع ثلاثة مبادئ جوهرية: "
        "أولاً: وصف السلعة أو المشروع بدقة تامة تمنع النزاع، بما يشمل المواصفات التقنية والجودة. "
        "ثانياً: تحديد الثمن الإجمالي والجدول الزمني للدفع بوضوح، مع جواز التأجيل والتقسيط. "
        "ثالثاً: تحمّل الصانع أو المقاول مخاطر التصنيع والتسليم حتى استلام المشروع. "
        "يُميّز المعيار الاستصناع عن السلم بأن السلم يشترط دفع الثمن كاملاً مقدماً، "
        "بينما يُجيز الاستصناع التأجيل."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-015", lang="en", domain="islamic_finance",
    task_type="executive_briefing", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="IFSB", source_url="https://www.ifsb.org/",
    scenario_brief=(
        "Task: Write a one-paragraph executive briefing on the global Islamic finance industry. "
        "Core facts: IFSB publishes annual stability reports on the industry. "
        "Required reasoning: Summarize size, geography, and key standards bodies. "
        "Expected knowledge: GCC and Malaysia dominance, IFSB and AAOIFI roles. "
        "Risk: Citing specific figures that may be outdated. "
        "Distinct from others: Executive briefing format requiring concise synthesis."
    ),
    prompt="Provide a one-paragraph executive briefing on the global Islamic finance industry for a non-specialist audience.",
    response=(
        "The global Islamic finance industry encompasses Islamic banking, sukuk capital markets, "
        "Islamic funds, and takaful insurance, with assets concentrated primarily in the GCC "
        "region, Malaysia, and Iran. The industry is governed by two principal international "
        "standards bodies: the Islamic Financial Services Board (IFSB), which focuses on "
        "prudential regulation and supervision, and the Accounting and Auditing Organization "
        "for Islamic Financial Institutions (AAOIFI), which sets Sharia and accounting standards. "
        "The industry has grown substantially over the past two decades, driven by strong demand "
        "in Muslim-majority markets and increasing interest from conventional financial centres. "
        "Accurate current size figures should be verified against the latest IFSB Islamic "
        "Financial Services Industry Stability Report."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-016", lang="ar", domain="islamic_finance",
    task_type="regulatory_explanation", difficulty="intermediate",
    country="Kuwait", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Central Bank of Kuwait", source_url="https://www.cbk.gov.kw/",
    scenario_brief=(
        "Task: Explain the regulatory framework for Islamic banks in Kuwait. "
        "Core facts: CBK regulates Islamic banks under a specific law. "
        "Required reasoning: Identify the key requirements: Sharia board, AAOIFI compliance, disclosure. "
        "Expected knowledge: CBK oversight, minimum Sharia board size, periodic disclosure. "
        "Risk: Describing the framework as identical to conventional bank regulation. "
        "Distinct from others: Kuwait-specific regulatory context."
    ),
    prompt="ما الإطار التنظيمي الذي يحكم البنوك الإسلامية في الكويت وما أبرز متطلباته؟",
    response=(
        "تخضع البنوك الإسلامية في الكويت لإشراف بنك الكويت المركزي وفق قانون تنظيم البنوك الإسلامية. "
        "يُلزم الإطار التنظيمي البنوكَ بتشكيل هيئات رقابة شرعية مستقلة لا تقل عن ثلاثة أعضاء مؤهلين، "
        "والالتزام بمعايير أيوفي أو المعايير المحلية المعتمدة من البنك المركزي. "
        "يشترط الإطار أيضاً الإفصاح الدوري عن الأداء الشرعي في التقارير السنوية. "
        "يُنصح بمراجعة أحدث التعاميم الصادرة عن البنك المركزي للتحقق من المتطلبات الحالية."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-017", lang="ar", domain="islamic_finance",
    task_type="regulatory_explanation", difficulty="intermediate",
    country="Saudi Arabia", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Saudi Central Bank SAMA", source_url="https://www.sama.gov.sa/",
    scenario_brief=(
        "Task: Explain SAMA's role in regulating Islamic finance in Saudi Arabia. "
        "Core facts: SAMA and CMA jointly regulate Islamic finance. "
        "Required reasoning: Distinguish SAMA's banking oversight from CMA's capital markets role. "
        "Expected knowledge: SAMA circulars, internal Sharia boards, product approval process. "
        "Risk: Confusing SAMA's role with AAOIFI's standard-setting role. "
        "Distinct from IF-TRAIN-016: Saudi Arabia-specific context with different regulatory structure."
    ),
    prompt="ما دور البنك المركزي السعودي ساما في تنظيم التمويل الإسلامي في المملكة؟",
    response=(
        "يُشرف البنك المركزي السعودي (ساما) على البنوك الإسلامية والتقليدية في المملكة العربية السعودية. "
        "تُلزم لوائح ساما البنوكَ بالالتزام بأحكام الشريعة في جميع منتجاتها وتشكيل هيئات رقابة شرعية داخلية. "
        "تُصدر ساما تعاميم دورية تُحدّد متطلبات الامتثال الشرعي وتُراجع المنتجات الجديدة قبل طرحها. "
        "تتولى هيئة السوق المالية (CMA) تنظيم صكوك الشركات والصناديق الإسلامية المدرجة. "
        "يُنصح بمراجعة الموقع الرسمي لساما للاطلاع على أحدث اللوائح والتعاميم."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-018", lang="ar", domain="islamic_finance",
    task_type="question_answering", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Explain the prohibition on riba in Islamic finance. "
        "Core facts: Riba is prohibited in the Quran and Sunnah. "
        "Required reasoning: Distinguish riba al-fadl from riba al-nasiah. "
        "Expected knowledge: Two types of riba, their application to banking. "
        "Risk: Treating all interest as identical without distinguishing the two types. "
        "Distinct from others: Foundational prohibition that underlies all Islamic finance."
    ),
    prompt="ما المقصود بتحريم الربا في التمويل الإسلامي وما أنواعه الرئيسية؟",
    response=(
        "الربا محرّم في القرآن الكريم والسنة النبوية وهو الزيادة المشروطة في القرض أو البيع. "
        "ينقسم الربا إلى نوعين رئيسيين: "
        "أولاً: ربا الفضل — وهو التفاضل في تبادل الأموال الربوية من جنس واحد كالذهب بالذهب أو الحنطة بالحنطة. "
        "ثانياً: ربا النسيئة — وهو الزيادة المشروطة في القرض مقابل تأجيل السداد، وهو الأكثر انتشاراً في المعاملات المصرفية. "
        "يُشكّل تحريم الربا الأساس الذي تقوم عليه جميع صيغ التمويل الإسلامي البديلة."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-019", lang="en", domain="islamic_finance",
    task_type="question_answering", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Explain the difference between Zakat and corporate tax. "
        "Core facts: Zakat is a religious obligation; corporate tax is a legal obligation. "
        "Required reasoning: Identify the calculation base, recipients, and legal nature of each. "
        "Expected knowledge: Zakat on net assets vs tax on income; nisab; eight Zakat categories. "
        "Risk: Treating Zakat as simply a form of tax. "
        "Distinct from others: Compares a religious obligation with a legal one."
    ),
    prompt="What is the difference between Zakat and conventional corporate tax in the GCC context?",
    response=(
        "Zakat is a religious obligation in Islam, calculated as 2.5% of eligible net assets "
        "held for one lunar year above the nisab (minimum threshold). It must be paid to the "
        "eight categories of recipients specified in the Quran. "
        "Conventional corporate tax is a government-imposed levy on business profits, calculated "
        "according to national tax law and payable to the state. "
        "Key differences: Zakat is based on wealth (net assets), not income; corporate tax is "
        "based on profit. Zakat recipients are religiously defined; tax revenue is allocated by "
        "government. In some GCC jurisdictions, Zakat may be credited against corporate tax "
        "obligations for qualifying entities. Specific rules require verification with qualified "
        "tax and Sharia advisors in each jurisdiction."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-020", lang="ar", domain="islamic_finance",
    task_type="question_answering", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Explain the prohibition on gharar in Islamic finance. "
        "Core facts: Gharar is excessive uncertainty that invalidates contracts. "
        "Required reasoning: Explain the degree of gharar that invalidates a contract. "
        "Expected knowledge: Minor vs major gharar; impact on derivatives and insurance. "
        "Risk: Treating all uncertainty as prohibited gharar. "
        "Distinct from IF-TRAIN-018: Focuses on gharar, not riba."
    ),
    prompt="ما المقصود بالغرر في العقود الإسلامية وكيف يُؤثر على صحة العقد؟",
    response=(
        "الغرر في الفقه الإسلامي هو الجهالة الفاحشة في أحد أركان العقد التي تُفضي إلى نزاع. "
        "ينقسم الغرر إلى يسير وهو المقبول في العقود، وفاحش وهو المُبطِل للعقد. "
        "يُعدّ الغرر فاحشاً إذا كان في أصل العقد أو ركنه الرئيسي كالثمن أو المبيع، "
        "أما إذا كان في أمر تابع فهو يسير ومعفو عنه. "
        "يُؤثر تحريم الغرر على المشتقات المالية التقليدية وبعض عقود التأمين التي تنطوي على جهالة فاحشة، "
        "مما يستدعي إعادة هيكلتها وفق الضوابط الشرعية."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-021", lang="ar", domain="islamic_finance",
    task_type="formal_correspondence", difficulty="intermediate",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write a formal Arabic letter requesting a Sukuk financing offer from a Kuwaiti Islamic bank. "
        "Core facts: The requester needs a Sukuk structure for an infrastructure project. "
        "Required reasoning: Include the key elements: structure, amount, timeline, Sharia opinion. "
        "Expected knowledge: Formal Arabic business letter structure; Sukuk-specific terminology. "
        "Risk: Omitting the request for a preliminary Sharia opinion. "
        "Distinct from others: Formal correspondence task with Sukuk-specific content."
    ),
    prompt="اكتب رسالة رسمية موجزة تطلب فيها من بنك إسلامي كويتي الحصول على عرض تمويل صكوك لمشروع بنية تحتية.",
    response=(
        "السادة / إدارة التمويل الهيكلي\nتحية طيبة وبعد،\n\n"
        "يسرّنا التواصل معكم بشأن الحصول على عرض تمويل بصيغة الصكوك لمشروع بنية تحتية. "
        "نودّ الاطلاع على الهيكل المقترح للإصدار، ومتطلبات الضمانات، والجدول الزمني المقترح، "
        "فضلاً عن الرأي الشرعي المبدئي من هيئة الرقابة الشرعية لديكم. "
        "يُرجى إرسال العرض الأولي إلى عنوان البريد الإلكتروني المُرفق خلال أسبوعين من تاريخه.\n\n"
        "مع خالص التقدير والاحترام،\n[الاسم] — [المسمى الوظيفي]"
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-022", lang="en", domain="islamic_finance",
    task_type="objection_handling", difficulty="intermediate",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Respond to the objection that Islamic finance products are just conventional loans with different names. "
        "Core facts: Structural differences between Murabaha and a conventional loan. "
        "Required reasoning: Identify three material structural differences. "
        "Expected knowledge: Asset ownership, risk sharing, Sharia board governance. "
        "Risk: Conceding the objection without explaining the structural differences. "
        "Distinct from others: Objection-handling format requiring persuasive response."
    ),
    prompt="A client argues that Islamic finance products are just conventional loans with different names. How should a banker respond to this objection?",
    response=(
        "This objection reflects a common misconception. The structural differences are material "
        "and legally significant. "
        "First, in Murabaha the bank genuinely owns the asset before selling it to the client, "
        "creating real commodity risk that does not exist in a conventional loan. "
        "Second, in Musharaka and Mudaraba the bank shares in business risk; conventional loans "
        "transfer all risk to the borrower regardless of outcome. "
        "Third, Islamic finance prohibits riba — a fixed predetermined return on money — "
        "requiring profit to be tied to real economic activity or asset performance. "
        "Fourth, Sharia board oversight provides an independent governance layer that reviews "
        "every product structure, which has no equivalent in conventional banking. "
        "These differences are not cosmetic; they affect legal rights, risk allocation, and "
        "the economic substance of the transaction."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-023", lang="ar", domain="islamic_finance",
    task_type="question_answering", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Explain the Wakala structure in Islamic asset management. "
        "Core facts: Wakala is an agency contract used in Sukuk and funds. "
        "Required reasoning: Explain the agent's obligations and the performance fee structure. "
        "Expected knowledge: Agent's fiduciary duty, incentive fee vs guaranteed return. "
        "Risk: Confusing Wakala with Mudaraba (different agency relationship). "
        "Distinct from others: Focuses on Wakala in asset management, not banking."
    ),
    prompt="ما هيكل الوكالة في إدارة الأصول الإسلامية وكيف تُحدَّد أتعاب الوكيل؟",
    response=(
        "الوكالة في إدارة الأصول الإسلامية عقد يُفوّض فيه المستثمر وكيلاً لإدارة أصوله وفق تعليمات محددة. "
        "يتحمل الوكيل واجب الأمانة والحرص في إدارة الأصول دون ضمان الربح. "
        "تُحدَّد أتعاب الوكيل عادةً بإحدى طريقتين: "
        "إما رسوم إدارة ثابتة كنسبة من الأصول المُدارة، "
        "أو رسوم أداء تُدفع عند تجاوز معدل عائد متفق عليه (Hurdle Rate). "
        "لا يجوز للوكيل ضمان عائد محدد للمستثمر لأن ذلك يُحوّل العقد إلى قرض بفائدة."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-024", lang="ar-en", domain="islamic_finance",
    task_type="structured_extraction", difficulty="intermediate",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Extract key terms from a fictional Ijarah Sukuk term sheet. "
        "Core facts: A fictional term sheet with specific structure, size, tenor, and rating. "
        "Required reasoning: Identify and tabulate the key terms in both Arabic and English. "
        "Expected knowledge: Sukuk terminology, bilingual financial table format. "
        "Risk: Omitting the Sharia certificate requirement. "
        "Distinct from IF-TRAIN-010: Extraction task from a specific fictional document."
    ),
    prompt=(
        "Extract the key terms from this fictional Ijarah Sukuk term sheet:\n"
        "Issuer: Gulf Infrastructure Holdings (fictional). Size: USD 300M. "
        "Tenor: 5 years. Structure: Ijarah. Profit rate: 3.75% p.a. "
        "Rating: BBB+. Governing law: English law with Sharia certificate."
    ),
    response=(
        "| المصطلح / Term | القيمة / Value |\n|---|---|\n"
        "| المُصدِر / Issuer | Gulf Infrastructure Holdings (fictional) |\n"
        "| حجم الإصدار / Size | USD 300M |\n"
        "| الأجل / Tenor | 5 years |\n"
        "| الهيكل / Structure | Ijarah Sukuk |\n"
        "| معدل العائد / Profit rate | 3.75% per annum |\n"
        "| التصنيف الائتماني / Rating | BBB+ |\n"
        "| القانون الحاكم / Governing law | English law + Sharia certificate |\n\n"
        "Note: All entities and figures are fictional and for training purposes only."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="IF-TRAIN-025", lang="ar", domain="islamic_finance",
    task_type="question_answering", difficulty="advanced",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="IFSB", source_url="https://www.ifsb.org/",
    scenario_brief=(
        "Task: Explain the Sharia governance framework for Islamic financial institutions. "
        "Core facts: IFSB-10 provides the governance standard. "
        "Required reasoning: Identify the three-tier governance structure. "
        "Expected knowledge: Internal Sharia board, external audit, regulatory oversight. "
        "Risk: Describing governance as only internal Sharia board oversight. "
        "Distinct from others: Focuses on the three-tier governance structure, not individual contracts."
    ),
    prompt="ما إطار الحوكمة الشرعية للمؤسسات المالية الإسلامية وفق معيار مجلس الخدمات المالية الإسلامية؟",
    response=(
        "يُحدّد معيار مجلس الخدمات المالية الإسلامية (IFSB-10) إطاراً ثلاثي المستويات لحوكمة الشريعة: "
        "المستوى الأول: هيئة الرقابة الشرعية الداخلية التي تُراجع المنتجات والعمليات وتُصدر الفتاوى. "
        "المستوى الثاني: المراجعة الشرعية الخارجية المستقلة التي تتحقق من تطبيق قرارات الهيئة الداخلية. "
        "المستوى الثالث: الرقابة الشرعية التنظيمية التي تُمارسها البنوك المركزية وهيئات الرقابة المالية. "
        "يُشترط في أعضاء هيئة الرقابة الشرعية الاستقلالية وعدم تعارض المصالح، "
        "ويُوصي المعيار بالحد الأدنى لعدد الأعضاء وآليات تعيينهم."
    ),
    factual_dep=True,
))

# ─── GCC Banking — Training ───────────────────────────────────────────────

TRAIN_EXAMPLES.append(make_example(
    family="GB-TRAIN-001", lang="en", domain="gcc_banking",
    task_type="regulatory_explanation", difficulty="intermediate",
    country="Kuwait", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Central Bank of Kuwait", source_url="https://www.cbk.gov.kw/",
    scenario_brief=(
        "Task: Explain Basel III capital adequacy requirements for Kuwaiti banks. "
        "Core facts: CBK implements Basel III with specific local thresholds. "
        "Required reasoning: List the four capital ratios and their minimum levels. "
        "Expected knowledge: CET1, Tier 1, total capital, conservation buffer. "
        "Risk: Citing Basel III minimums without the CBK-specific conservation buffer. "
        "Distinct from others: Kuwait-specific Basel III implementation."
    ),
    prompt="What are the primary Basel III capital adequacy requirements for commercial banks in Kuwait?",
    response=(
        "Under Basel III as implemented by the Central Bank of Kuwait, commercial banks must maintain: "
        "(1) A Common Equity Tier 1 (CET1) ratio of at least 4.5% of risk-weighted assets; "
        "(2) A Tier 1 capital ratio of at least 6% of risk-weighted assets; "
        "(3) A total capital ratio of at least 8% of risk-weighted assets; and "
        "(4) A capital conservation buffer of 2.5% of risk-weighted assets, bringing the "
        "effective CET1 minimum to 7%. "
        "The CBK may impose additional capital surcharges for systemically important institutions. "
        "Exact current thresholds should be verified against the latest CBK circulars."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="GB-TRAIN-002", lang="ar", domain="gcc_banking",
    task_type="question_answering", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Basel Committee on Banking Supervision", source_url="https://www.bis.org/bcbs/",
    scenario_brief=(
        "Task: Explain the difference between Tier 1 and Tier 2 capital. "
        "Core facts: Basel III defines two tiers of regulatory capital. "
        "Required reasoning: Explain why Tier 1 is more loss-absorbing than Tier 2. "
        "Expected knowledge: Common equity, retained earnings, subordinated debt. "
        "Risk: Treating both tiers as equivalent in loss absorption. "
        "Distinct from GB-TRAIN-001: Focuses on the capital tiers, not the ratios."
    ),
    prompt="ما الفرق بين رأس المال من الشريحة الأولى والشريحة الثانية في إطار بازل الثالث؟",
    response=(
        "رأس المال من الشريحة الأولى هو رأس المال الأساسي للبنك ويشمل حقوق المساهمين العادية والأرباح المحتجزة. "
        "يُعدّ الأكثر قدرة على استيعاب الخسائر لأنه متاح دائماً دون قيود أو استحقاق. "
        "أما رأس المال من الشريحة الثانية فيشمل الديون الثانوية وبعض الاحتياطيات الخاصة، "
        "وهو أقل قدرة على استيعاب الخسائر لأنه يستحق السداد ولا يُمتص إلا عند الإفلاس. "
        "لذلك تشترط بازل الثالث نسبة أعلى من الشريحة الأولى مقارنةً بالشريحة الثانية."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="GB-TRAIN-003", lang="en", domain="gcc_banking",
    task_type="risk_identification", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify credit risks in a syndicated loan to a fictional GCC petrochemical company. "
        "Core facts: Fictional company with commodity price exposure and concentrated offtake. "
        "Required reasoning: Identify five distinct credit risk categories. "
        "Expected knowledge: Commodity risk, offtake concentration, refinancing, FX, regulatory. "
        "Risk: Listing generic risks without connecting them to the specific petrochemical context. "
        "Distinct from others: Advanced risk identification for a specific fictional industry."
    ),
    prompt="Identify the key credit risks in extending a syndicated loan to a fictional GCC petrochemical company with concentrated offtake and USD revenue.",
    response=(
        "Key credit risks for this fictional GCC petrochemical syndicated loan: "
        "(1) Commodity price risk: Revenue is tied to hydrocarbon feedstock and product prices, "
        "which are cyclical and can compress margins rapidly. "
        "(2) Offtake concentration: Dependence on a small number of buyers creates counterparty "
        "concentration risk; loss of a single major buyer could impair debt service. "
        "(3) Refinancing risk: Large syndicated facilities may face rollover challenges in "
        "tightening credit markets, particularly if the company's credit profile deteriorates. "
        "(4) Regulatory and environmental risk: Evolving GCC environmental standards may impose "
        "significant capital expenditure obligations not reflected in current projections. "
        "(5) Currency mismatch: If operating costs include significant local-currency components "
        "while revenues are USD-denominated, margin compression occurs during periods of "
        "local currency appreciation. "
        "Note: This analysis uses a fictional company for training purposes."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="GB-TRAIN-004", lang="ar", domain="gcc_banking",
    task_type="document_summarization", difficulty="intermediate",
    country="UAE", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="UAE Central Bank", source_url="https://www.centralbank.ae/",
    scenario_brief=(
        "Task: Summarize AML requirements for UAE banks. "
        "Core facts: UAE Central Bank AML framework based on FATF standards. "
        "Required reasoning: Identify the four key AML obligations. "
        "Expected knowledge: KYC, transaction monitoring, record keeping, compliance officer. "
        "Risk: Omitting the record retention period. "
        "Distinct from GB-TRAIN-001: AML compliance, not capital adequacy."
    ),
    prompt="لخّص المتطلبات الرئيسية لمكافحة غسل الأموال المفروضة على البنوك في الإمارات العربية المتحدة.",
    response=(
        "تُلزم متطلبات مكافحة غسل الأموال في الإمارات البنوكَ بأربعة التزامات رئيسية: "
        "أولاً: تطبيق إجراءات العناية الواجبة بالعميل (KYC) عند فتح الحسابات وبصفة دورية لتحديث بيانات العملاء. "
        "ثانياً: مراقبة المعاملات بصفة مستمرة والإبلاغ عن المشبوهة لوحدة المعلومات المالية (FIU). "
        "ثالثاً: الاحتفاظ بسجلات المعاملات والعملاء لمدة لا تقل عن خمس سنوات. "
        "رابعاً: تعيين مسؤول امتثال مؤهل ومعتمد يتولى الإشراف على برنامج مكافحة غسل الأموال. "
        "يُنصح بمراجعة أحدث التعاميم الصادرة عن المصرف المركزي للتحقق من المتطلبات الحالية."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="GB-TRAIN-005", lang="en", domain="gcc_banking",
    task_type="hallucination_resistance", difficulty="basic",
    country="Kuwait", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to provide the current CBK discount rate. "
        "Core facts: Real-time rate data is not available. "
        "Required reasoning: Direct the user to the official CBK source. "
        "Expected knowledge: CBK website URL; risk of misleading with invented rate. "
        "Risk: Providing a plausible-sounding but invented rate. "
        "Distinct from IF-TRAIN-012: Banking rate refusal, not Islamic finance rate."
    ),
    prompt="What is the current discount rate set by the Central Bank of Kuwait?",
    response=(
        "I do not have access to real-time data and cannot provide the current discount rate "
        "set by the Central Bank of Kuwait. "
        "Rates are subject to change and should be verified directly on the CBK's official "
        "website at www.cbk.gov.kw or through a licensed financial data provider. "
        "Providing an invented rate could mislead financial or investment decisions."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="GB-TRAIN-006", lang="ar", domain="gcc_banking",
    task_type="structured_extraction", difficulty="intermediate",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Extract key information from a fictional credit report summary. "
        "Core facts: Fictional company with specific financial ratios and credit rating. "
        "Required reasoning: Identify and tabulate the five key credit metrics. "
        "Expected knowledge: Total assets, debt-to-equity, credit rating, net profit. "
        "Risk: Omitting the debt-to-equity ratio which is the key leverage indicator. "
        "Distinct from IF-TRAIN-024: Banking credit metrics, not Sukuk terms."
    ),
    prompt=(
        "استخرج المعلومات الرئيسية من ملخص التقرير الائتماني الافتراضي التالي:\n"
        "الشركة: مجموعة الخليج للصناعات (افتراضية). إجمالي الأصول: 450 مليون دولار. "
        "نسبة الدين إلى حقوق الملكية: 1.8. التصنيف الائتماني: BBB. صافي الربح: 22 مليون دولار."
    ),
    response=(
        "| البند | القيمة |\n|---|---|\n"
        "| اسم الشركة | مجموعة الخليج للصناعات (افتراضية) |\n"
        "| إجمالي الأصول | 450 مليون دولار |\n"
        "| نسبة الدين إلى حقوق الملكية | 1.8 |\n"
        "| التصنيف الائتماني | BBB |\n"
        "| صافي الربح | 22 مليون دولار |\n\n"
        "ملاحظة: الشركة افتراضية لأغراض التدريب فقط."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="GB-TRAIN-007", lang="ar", domain="gcc_banking",
    task_type="formal_correspondence", difficulty="intermediate",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Draft a formal Arabic letter requesting a credit facility review. "
        "Core facts: Company wants to discuss increasing its credit limits. "
        "Required reasoning: Include the key elements: credit history, transaction volume, meeting request. "
        "Expected knowledge: Formal Arabic banking correspondence structure. "
        "Risk: Omitting the request for a specific meeting date. "
        "Distinct from IF-TRAIN-021: Banking credit facility, not Sukuk financing."
    ),
    prompt="اكتب رسالة رسمية باللغة العربية تطلب فيها من بنك تجاري كويتي مراجعة التسهيلات الائتمانية الممنوحة لشركتك.",
    response=(
        "السادة / إدارة الائتمان\nتحية طيبة وبعد،\n\n"
        "نتشرف بتقديم طلبنا لمراجعة التسهيلات الائتمانية الممنوحة لشركتنا في ضوء التطورات الأخيرة في أعمالنا. "
        "نأمل أن تأخذوا بعين الاعتبار سجلنا الائتماني الجيد وحجم معاملاتنا المصرفية معكم خلال السنوات الماضية. "
        "نطلب تحديد موعد مناسب لمناقشة إمكانية تعديل حدود التسهيلات الحالية وفق احتياجاتنا التشغيلية.\n\n"
        "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي]"
    ),
))

# ─── Telecommunications — Training ────────────────────────────────────────

TRAIN_EXAMPLES.append(make_example(
    family="TC-TRAIN-001", lang="en", domain="telecommunications",
    task_type="regulatory_explanation", difficulty="intermediate",
    country="Kuwait", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="CITRA Kuwait", source_url="https://www.citra.gov.kw/",
    scenario_brief=(
        "Task: Explain the key regulatory obligations for mobile network operators in Kuwait. "
        "Core facts: CITRA regulates MNOs in Kuwait. "
        "Required reasoning: List five distinct regulatory obligations. "
        "Expected knowledge: Spectrum licences, QoS, lawful interception, number portability, Kuwaitisation. "
        "Risk: Omitting the lawful interception requirement. "
        "Distinct from others: Kuwait-specific telecom regulation."
    ),
    prompt="What are the key regulatory obligations for mobile network operators in Kuwait under CITRA?",
    response=(
        "Mobile network operators in Kuwait are regulated by the Communications and Information "
        "Technology Regulatory Authority (CITRA). Key obligations include: "
        "(1) Obtaining and maintaining valid spectrum licences for each frequency band used; "
        "(2) Meeting published Quality of Service (QoS) benchmarks for network availability, "
        "call drop rates, and data speeds; "
        "(3) Complying with lawful interception requirements to support law enforcement; "
        "(4) Adhering to number portability regulations enabling subscribers to retain their "
        "numbers when switching operators; and "
        "(5) Meeting local content and Kuwaitisation employment commitments. "
        "Specific current requirements should be verified against CITRA's published regulations."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="TC-TRAIN-002", lang="ar", domain="telecommunications",
    task_type="question_answering", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="ITU", source_url="https://www.itu.int/",
    scenario_brief=(
        "Task: Explain QoS metrics in telecommunications networks. "
        "Core facts: ITU defines standard QoS metrics. "
        "Required reasoning: Identify and define the four main QoS metrics. "
        "Expected knowledge: Latency, packet loss, bandwidth, jitter. "
        "Risk: Confusing latency with jitter. "
        "Distinct from TC-TRAIN-001: Focuses on technical QoS metrics, not regulatory obligations."
    ),
    prompt="ما المقصود بجودة الخدمة في شبكات الاتصالات وما مؤشراتها التقنية الرئيسية؟",
    response=(
        "جودة الخدمة (QoS) في شبكات الاتصالات هي مجموعة من المعايير التقنية التي تضمن أداء الشبكة وفق مستويات محددة. "
        "تشمل مؤشراتها الرئيسية: "
        "أولاً: زمن الاستجابة (Latency) — الوقت المستغرق لانتقال البيانات من المُرسِل إلى المُستقبِل. "
        "ثانياً: معدل فقدان الحزم (Packet Loss) — نسبة الحزم التي لا تصل إلى وجهتها. "
        "ثالثاً: عرض النطاق الترددي (Bandwidth) — الحد الأقصى لمعدل نقل البيانات. "
        "رابعاً: التذبذب (Jitter) — التفاوت في زمن وصول الحزم المتتالية، وهو مختلف عن زمن الاستجابة."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="TC-TRAIN-003", lang="en", domain="telecommunications",
    task_type="risk_identification", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify commercial risks for a fictional GCC telecom expanding into enterprise managed services. "
        "Core facts: Fictional MNO with legacy infrastructure entering enterprise IT services. "
        "Required reasoning: Identify five distinct commercial risks specific to this expansion. "
        "Expected knowledge: Margin compression, integration complexity, competition, talent, concentration. "
        "Risk: Listing generic IT risks without connecting them to the telecom context. "
        "Distinct from GB-TRAIN-003: Telecom-specific commercial risks, not banking credit risks."
    ),
    prompt="Identify the main commercial risks for a fictional GCC telecom operator expanding into enterprise managed IT services.",
    response=(
        "Key commercial risks for a fictional GCC telecom expanding into enterprise managed services: "
        "(1) Margin compression: Enterprise managed services require significant upfront investment "
        "in skilled personnel and infrastructure, with longer sales cycles than consumer services. "
        "(2) Integration complexity: Connecting legacy telecom network infrastructure with "
        "cloud-native enterprise platforms creates technical debt and delivery risk. "
        "(3) Competition from global system integrators: Established players such as Accenture, "
        "IBM, and regional integrators may outbid on large enterprise contracts. "
        "(4) Talent scarcity: Skilled enterprise IT architects and project managers are in short "
        "supply across the GCC, limiting delivery capacity. "
        "(5) Revenue concentration: Dependence on a small number of government enterprise clients "
        "creates revenue volatility if contract renewals are delayed or lost. "
        "Note: This analysis uses a fictional operator for training purposes."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="TC-TRAIN-004", lang="ar", domain="telecommunications",
    task_type="executive_briefing", difficulty="intermediate",
    country="Kuwait", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write an executive briefing on 5G's impact on the Kuwaiti telecom sector. "
        "Core facts: 5G enables IoT and digital transformation but requires infrastructure investment. "
        "Required reasoning: Identify opportunities and challenges specific to Kuwait. "
        "Expected knowledge: IoT, smart cities, infrastructure costs, spectrum availability. "
        "Risk: Describing 5G benefits without addressing the investment and spectrum challenges. "
        "Distinct from TC-TRAIN-001: Strategic briefing, not regulatory obligations."
    ),
    prompt="أعدّ موجزاً تنفيذياً حول تأثير تقنية الجيل الخامس على قطاع الاتصالات في الكويت.",
    response=(
        "تُتيح تقنية الجيل الخامس في الكويت فرصاً استراتيجية لتعزيز الاتصال بين الأجهزة الذكية "
        "ودعم التحول الرقمي في القطاعين الحكومي والصناعي. "
        "تشمل الفرص الرئيسية: تطوير تطبيقات المدن الذكية، وتحسين كفاءة الموانئ والمطارات، "
        "وتمكين الرعاية الصحية عن بُعد. "
        "في المقابل، تواجه شركات الاتصالات الكويتية تحديات تتعلق بتكاليف ترقية البنية التحتية "
        "وتوافر الطيف الترددي المناسب وتحقيق عائد استثماري كافٍ في سوق محدود الحجم. "
        "ملاحظة: هذا تحليل افتراضي لأغراض التدريب."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="TC-TRAIN-005", lang="ar-en", domain="telecommunications",
    task_type="arabic_english_translation", difficulty="intermediate",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'Spectrum Auction' into Arabic with a bilingual explanation. "
        "Core facts: Spectrum auctions are used by regulators to allocate frequency bands. "
        "Required reasoning: Explain the competitive process and its purpose. "
        "Expected knowledge: Frequency allocation, licence fees, competitive bidding. "
        "Risk: Translating 'auction' without explaining the regulatory context. "
        "Distinct from IF-TRAIN-009: Telecom terminology, not Islamic finance."
    ),
    prompt="Translate the telecom regulatory term 'Spectrum Auction' into Arabic and provide a bilingual explanation of how it works.",
    response=(
        "الترجمة: مزاد الطيف الترددي.\n\n"
        "الشرح بالعربية: مزاد الطيف الترددي هو عملية تنافسية تُجريها هيئات تنظيم الاتصالات "
        "لتخصيص نطاقات ترددية لمشغلي الشبكات مقابل رسوم ترخيص، مما يضمن التوزيع الكفء للطيف الشحيح.\n\n"
        "English explanation: A spectrum auction is a competitive process conducted by telecom "
        "regulators to allocate specific frequency bands to network operators in exchange for "
        "licence fees. Operators bid for the right to use designated spectrum for a defined period. "
        "This mechanism ensures efficient allocation of scarce radio spectrum and generates "
        "government revenue."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="TC-TRAIN-006", lang="en", domain="telecommunications",
    task_type="hallucination_resistance", difficulty="basic",
    country="Kuwait", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to provide current 5G coverage percentage for Kuwait. "
        "Core facts: Coverage data changes frequently and is not available in real-time. "
        "Required reasoning: Direct the user to the operators' official sources. "
        "Expected knowledge: Three Kuwaiti MNOs; CITRA as regulator. "
        "Risk: Inventing a coverage percentage. "
        "Distinct from GB-TRAIN-005: Telecom coverage data, not banking rates."
    ),
    prompt="What is the current 5G network coverage percentage in Kuwait?",
    response=(
        "I do not have access to real-time network coverage data and cannot provide a current "
        "5G coverage percentage for Kuwait. Coverage figures change frequently as operators "
        "expand their networks. "
        "For accurate, up-to-date information, please consult the official websites of "
        "Zain Kuwait, Ooredoo Kuwait, and STC Kuwait, or the CITRA regulator. "
        "Providing an invented figure could mislead infrastructure or investment decisions."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="TC-TRAIN-007", lang="ar", domain="telecommunications",
    task_type="regulatory_explanation", difficulty="advanced",
    country="UAE", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="UAE TRA", source_url="https://www.tra.gov.ae/",
    scenario_brief=(
        "Task: Explain the UAE telecom regulatory framework under TRA. "
        "Core facts: TRA regulates spectrum, licences, and QoS in the UAE. "
        "Required reasoning: Identify TRA's three main regulatory functions. "
        "Expected knowledge: Licensing, spectrum management, consumer protection. "
        "Risk: Confusing TRA's role with the operators' commercial decisions. "
        "Distinct from TC-TRAIN-001: UAE-specific framework, not Kuwait."
    ),
    prompt="ما الإطار التنظيمي الذي يحكم قطاع الاتصالات في الإمارات العربية المتحدة وما دور هيئة تنظيم الاتصالات؟",
    response=(
        "تُنظّم هيئة تنظيم الاتصالات والحكومة الرقمية (TRA) قطاع الاتصالات في الإمارات. "
        "تضطلع الهيئة بثلاث وظائف تنظيمية رئيسية: "
        "أولاً: منح التراخيص لمشغلي الشبكات وتجديدها وفق شروط محددة. "
        "ثانياً: إدارة الطيف الترددي وتخصيصه بين المشغلين بما يضمن الاستخدام الأمثل. "
        "ثالثاً: حماية المستهلك من خلال تحديد معايير جودة الخدمة ومعالجة شكاوى المشتركين. "
        "يُلزم الإطار التنظيمي المشغلين بالامتثال لمتطلبات الاعتراض القانوني وحماية بيانات المستهلكين. "
        "يُنصح بمراجعة الموقع الرسمي للهيئة للاطلاع على أحدث اللوائح."
    ),
    factual_dep=True,
))

# ─── Government Regulation — Training ─────────────────────────────────────

TRAIN_EXAMPLES.append(make_example(
    family="GR-TRAIN-001", lang="ar", domain="government_regulation",
    task_type="regulatory_explanation", difficulty="intermediate",
    country="Kuwait", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Kuwait Central Tenders Committee", source_url="https://www.ctc.gov.kw/",
    scenario_brief=(
        "Task: Explain Kuwait's government procurement regulatory framework. "
        "Core facts: Kuwait's Public Tenders Law governs government procurement. "
        "Required reasoning: Identify the three key requirements: public tender, financial threshold, Kuwaitisation. "
        "Expected knowledge: CTC role, tender thresholds, local preference rules. "
        "Risk: Omitting the local preference requirement. "
        "Distinct from others: Kuwait-specific procurement regulation."
    ),
    prompt="ما الإطار التنظيمي الذي يحكم إجراءات المشتريات الحكومية في الكويت؟",
    response=(
        "تخضع المشتريات الحكومية في الكويت لأحكام قانون المناقصات العامة وتعديلاته، "
        "الذي تُشرف على تطبيقه الجهاز المركزي للمناقصات العامة. "
        "يُلزم القانون الجهات الحكومية بطرح المناقصات العامة للعقود التي تتجاوز حدوداً مالية محددة، "
        "مع منح أولوية للمنتجات والشركات الكويتية في بعض الحالات وفق نظام التفضيل المحلي. "
        "يُنصح بمراجعة النص الرسمي للقانون والتعاميم الصادرة عن الجهاز للاطلاع على الإجراءات المحدّثة."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="GR-TRAIN-002", lang="en", domain="government_regulation",
    task_type="document_summarization", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Saudi PDPL", source_url="https://www.pdpl.gov.sa/",
    scenario_brief=(
        "Task: Summarize key principles of GCC data protection frameworks. "
        "Core facts: Saudi PDPL, UAE Federal DPL, and Kuwait draft law. "
        "Required reasoning: Identify five common principles across GCC jurisdictions. "
        "Expected knowledge: Lawful basis, purpose limitation, data minimisation, subject rights, transfer restrictions. "
        "Risk: Treating all GCC data protection laws as identical. "
        "Distinct from GR-TRAIN-001: Data protection, not procurement."
    ),
    prompt="Summarise the key principles common to GCC data protection frameworks.",
    response=(
        "GCC data protection frameworks share several common principles derived from international "
        "best practices: "
        "(1) Lawful basis: Personal data must be processed on a recognised lawful basis such as "
        "consent, contract, or legitimate interest. "
        "(2) Purpose limitation: Data collected for a specific purpose must not be used for "
        "incompatible purposes without additional consent. "
        "(3) Data minimisation: Only data necessary for the stated purpose should be collected. "
        "(4) Data subject rights: Individuals have rights to access, correct, and in some "
        "jurisdictions delete their personal data. "
        "(5) Cross-border transfer restrictions: Transfers to countries without adequate "
        "protection require specific safeguards. "
        "Saudi Arabia's PDPL, the UAE's Federal DPL, and Kuwait's draft data protection law "
        "each have jurisdiction-specific provisions. Legal counsel should be consulted for "
        "compliance in each country."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="GR-TRAIN-003", lang="ar", domain="government_regulation",
    task_type="question_answering", difficulty="basic",
    country="Kuwait", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Kuwait Audit Bureau", source_url="https://www.audit.gov.kw/",
    scenario_brief=(
        "Task: Explain the role of Kuwait's Audit Bureau. "
        "Core facts: The Audit Bureau is Kuwait's supreme audit institution. "
        "Required reasoning: Identify the Bureau's mandate and reporting line. "
        "Expected knowledge: Financial audit of government entities, reporting to parliament. "
        "Risk: Confusing the Audit Bureau with the Ministry of Finance. "
        "Distinct from GR-TRAIN-001: Audit oversight, not procurement regulation."
    ),
    prompt="ما دور ديوان المحاسبة في الكويت وما صلاحياته الرقابية على الإنفاق العام؟",
    response=(
        "ديوان المحاسبة في الكويت هو الجهاز الرقابي المالي الأعلى في الدولة، "
        "ويختص بمراجعة الحسابات الختامية للجهات الحكومية والتحقق من صحة الإنفاق العام ومدى مطابقته للقوانين. "
        "يرفع الديوان تقاريره السنوية إلى مجلس الأمة مباشرةً، مما يُعزز استقلاليته عن السلطة التنفيذية. "
        "يُسهم الديوان في تعزيز الشفافية والمساءلة في إدارة المال العام من خلال تدقيق العقود الحكومية الكبرى. "
        "تستند هذه المعلومات إلى البيانات المتاحة للعموم ولا تُعدّ تفسيراً قانونياً رسمياً."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="GR-TRAIN-004", lang="en", domain="government_regulation",
    task_type="risk_identification", difficulty="advanced",
    country="Kuwait", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify regulatory risks for a fictional foreign company entering Kuwait. "
        "Core facts: Fictional foreign company seeking to establish a branch. "
        "Required reasoning: Identify five distinct regulatory risk categories. "
        "Expected knowledge: Commercial agency, foreign ownership, Kuwaitisation, licensing, dispute resolution. "
        "Risk: Omitting the commercial agency requirement which is specific to Kuwait. "
        "Distinct from GR-TRAIN-001: Entry risks, not procurement regulation."
    ),
    prompt="What are the key regulatory risks for a fictional foreign company establishing a branch in Kuwait?",
    response=(
        "Key regulatory risks for a fictional foreign company entering Kuwait: "
        "(1) Commercial agency requirements: Certain commercial activities require a Kuwaiti "
        "commercial agent, which creates dependency and potential conflicts of interest. "
        "(2) Foreign ownership restrictions: Some sectors restrict or prohibit foreign equity "
        "participation, requiring joint venture structures with Kuwaiti partners. "
        "(3) Labour nationalisation quotas: Companies must meet prescribed Kuwaiti employee "
        "ratios (Kuwaitisation), which can constrain hiring flexibility. "
        "(4) Regulatory approval timelines: Licensing processes can be lengthy and unpredictable, "
        "delaying market entry. "
        "(5) Dispute resolution: Commercial disputes may be subject to Kuwaiti courts unless "
        "international arbitration is explicitly agreed in contracts. "
        "Legal counsel familiar with Kuwaiti commercial law should be engaged before market entry. "
        "Note: This analysis uses a fictional company for training purposes."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="GR-TRAIN-005", lang="ar", domain="government_regulation",
    task_type="executive_briefing", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Saudi Vision 2030", source_url="https://www.vision2030.gov.sa/",
    scenario_brief=(
        "Task: Brief on Saudi Vision 2030's impact on GCC business environment. "
        "Core facts: Vision 2030 aims to diversify Saudi economy away from oil. "
        "Required reasoning: Identify three sectors being developed and their GCC impact. "
        "Expected knowledge: Tourism, entertainment, financial services; regional competition. "
        "Risk: Describing Vision 2030 as only affecting Saudi Arabia. "
        "Distinct from GR-TRAIN-001: Strategic policy briefing, not regulatory compliance."
    ),
    prompt="أعدّ موجزاً تنفيذياً حول أثر رؤية المملكة العربية السعودية 2030 على بيئة الأعمال في منطقة الخليج.",
    response=(
        "تسعى رؤية المملكة العربية السعودية 2030 إلى تنويع الاقتصاد الوطني وتقليل الاعتماد على النفط "
        "من خلال تطوير قطاعات السياحة والترفيه والخدمات المالية والصناعة التحويلية. "
        "تنعكس هذه التحولات على دول الخليج المجاورة من خلال ثلاثة محاور: "
        "أولاً: تصاعد التنافسية الإقليمية على جذب الاستثمار الأجنبي المباشر. "
        "ثانياً: فرص التكامل الاقتصادي في سلاسل التوريد والخدمات اللوجستية. "
        "ثالثاً: الضغط على الكوادر البشرية الماهرة في ظل توسع سوق العمل السعودي. "
        "تستند هذه المعلومات إلى الموقع الرسمي لرؤية 2030 وتعكس التوجهات العامة المعلنة."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="GR-TRAIN-006", lang="ar-en", domain="government_regulation",
    task_type="arabic_english_translation", difficulty="intermediate",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'Regulatory Sandbox' into Arabic with a bilingual explanation. "
        "Core facts: Regulatory sandboxes are used by GCC regulators for fintech innovation. "
        "Required reasoning: Explain the controlled testing environment concept. "
        "Expected knowledge: Relaxed requirements, defined period, regulatory supervision. "
        "Risk: Translating 'sandbox' literally without the regulatory context. "
        "Distinct from TC-TRAIN-005: Government regulation context, not telecom."
    ),
    prompt="Translate the regulatory term 'Regulatory Sandbox' into Arabic and explain how it works in the GCC context.",
    response=(
        "الترجمة: البيئة التنظيمية التجريبية (أو الحاضنة التنظيمية).\n\n"
        "الشرح بالعربية: إطار تنظيمي يُتيح للشركات المبتكرة اختبار منتجاتها وخدماتها في بيئة محكومة "
        "تحت إشراف الجهة التنظيمية، مع تخفيف بعض الاشتراطات التنظيمية لفترة زمنية محددة.\n\n"
        "English explanation: A regulatory sandbox is a framework allowing innovators to test "
        "products and services in a controlled environment under regulatory supervision, with "
        "some standard requirements relaxed for a defined period to encourage innovation. "
        "Several GCC regulators including the CBK, SAMA, and CBUAE have established sandbox "
        "programmes for fintech companies."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="GR-TRAIN-007", lang="en", domain="government_regulation",
    task_type="hallucination_resistance", difficulty="basic",
    country="Kuwait", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to provide exact penalties under Kuwait's draft data protection law. "
        "Core facts: Kuwait's data protection law was not fully enacted as of knowledge cutoff. "
        "Required reasoning: Explain why exact penalties cannot be provided for a draft law. "
        "Expected knowledge: CITRA as the relevant authority; need for legal counsel. "
        "Risk: Inventing penalty figures for a law that may not yet be enacted. "
        "Distinct from GR-TRAIN-002: Refusal task about a specific jurisdiction's law."
    ),
    prompt="What is the exact penalty for a company that violates Kuwait's data protection law?",
    response=(
        "Kuwait's comprehensive data protection law was still in draft form as of my knowledge "
        "cutoff; I cannot provide exact penalty figures for a law that may not yet be fully enacted. "
        "The applicable penalties and enforcement mechanisms depend on the final enacted text. "
        "For authoritative and current information, please consult CITRA or a licensed Kuwaiti "
        "legal practitioner who can advise on the current status and applicable requirements. "
        "Providing invented penalty figures could mislead compliance decisions."
    ),
))

# ─── Energy and Logistics — Training ──────────────────────────────────────

TRAIN_EXAMPLES.append(make_example(
    family="EL-TRAIN-001", lang="en", domain="energy_logistics",
    task_type="executive_briefing", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Qatar Energy", source_url="https://www.qatarenergy.qa/",
    scenario_brief=(
        "Task: Brief on the strategic importance of GCC LNG export capacity. "
        "Core facts: Qatar is one of the world's largest LNG exporters. "
        "Required reasoning: Explain the link between LNG exports and GCC economic diversification. "
        "Expected knowledge: North Field expansion, global energy security, sovereign wealth. "
        "Risk: Citing specific production volumes that may be outdated. "
        "Distinct from others: Energy sector strategic briefing."
    ),
    prompt="Provide a brief executive briefing on the strategic importance of the GCC's LNG export capacity for global energy security.",
    response=(
        "The GCC region, led by Qatar as one of the world's largest LNG exporters, plays a "
        "critical role in global energy security by providing long-term supply contracts to "
        "European and Asian buyers. Qatar's North Field expansion programme is expected to "
        "increase global LNG supply capacity significantly over the coming decade. "
        "For GCC governments, LNG export revenue provides a more stable income stream than "
        "crude oil because of the long-term nature of LNG supply contracts. "
        "This revenue supports sovereign wealth fund accumulation and economic diversification. "
        "Specific production volumes and expansion timelines should be verified against current "
        "data from Qatar Energy or the International Energy Agency."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="EL-TRAIN-002", lang="ar", domain="energy_logistics",
    task_type="question_answering", difficulty="basic",
    country="Kuwait", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Kuwait Petroleum Corporation", source_url="https://www.kpc.com.kw/",
    scenario_brief=(
        "Task: Explain KPC's role in Kuwait's energy sector. "
        "Core facts: KPC is the state holding company for oil and gas. "
        "Required reasoning: Identify KPC's main subsidiaries and their functions. "
        "Expected knowledge: Exploration, refining, distribution, international marketing. "
        "Risk: Confusing KPC with Kuwait Oil Company (KOC). "
        "Distinct from EL-TRAIN-001: Kuwait-specific energy sector, not GCC LNG."
    ),
    prompt="ما دور مؤسسة البترول الكويتية في قطاع الطاقة الكويتي وما شركاتها التابعة الرئيسية؟",
    response=(
        "مؤسسة البترول الكويتية (KPC) هي الشركة القابضة الحكومية المسؤولة عن إدارة قطاع النفط والغاز في الكويت. "
        "تُدير المؤسسة سلسلة القيمة الكاملة من خلال شركاتها التابعة الرئيسية: "
        "شركة نفط الكويت (KOC) للاستكشاف والإنتاج، "
        "وشركة البترول الوطنية الكويتية (KNPC) للتكرير، "
        "وشركة الناقلات البترولية الكويتية (KOTC) للنقل البحري، "
        "وشركة البترول الكويتية الدولية (KPI) للتسويق الدولي. "
        "تُعدّ المؤسسة ركيزة أساسية في الاقتصاد الكويتي وتُسهم في تمويل الميزانية العامة للدولة. "
        "تستند هذه المعلومات إلى البيانات المتاحة للعموم."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="EL-TRAIN-003", lang="en", domain="energy_logistics",
    task_type="risk_identification", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify supply chain risks for a fictional GCC petrochemical exporter to Asia. "
        "Core facts: Fictional company with shipping lane exposure and demand volatility. "
        "Required reasoning: Identify five distinct supply chain risk categories. "
        "Expected knowledge: Strait of Hormuz, Chinese demand cycles, port congestion, feedstock, regulatory. "
        "Risk: Listing generic supply chain risks without the GCC-Asia specific context. "
        "Distinct from GB-TRAIN-003: Supply chain risks, not credit risks."
    ),
    prompt="What are the primary supply chain risks for a fictional GCC petrochemical company exporting to Asian markets?",
    response=(
        "Primary supply chain risks for a fictional GCC petrochemical exporter to Asia: "
        "(1) Shipping lane disruption: Geopolitical tensions in the Strait of Hormuz or the "
        "Red Sea can delay tanker voyages and increase insurance costs. "
        "(2) Demand volatility: Asian petrochemical demand is sensitive to Chinese economic "
        "cycles; a slowdown in Chinese manufacturing reduces offtake volumes. "
        "(3) Port congestion: Major Asian ports periodically experience significant delays "
        "that increase inventory carrying costs and disrupt delivery schedules. "
        "(4) Feedstock price risk: Naphtha and ethane price movements affect production "
        "economics and can compress margins when product prices do not adjust proportionally. "
        "(5) Regulatory risk: Evolving environmental standards in importing countries may "
        "require product reformulation or certification, adding cost and lead time. "
        "Note: This analysis uses a fictional company for training purposes."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="EL-TRAIN-004", lang="ar", domain="energy_logistics",
    task_type="document_summarization", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="GCC Ports Authorities", source_url="https://www.moci.gov.kw/",
    scenario_brief=(
        "Task: Summarize the importance of major GCC ports in regional supply chains. "
        "Core facts: Jebel Ali, King Abdullah Port, and Shuwaikh are key GCC ports. "
        "Required reasoning: Explain the three functions: logistics hub, free zones, transit. "
        "Expected knowledge: Port infrastructure, free zone benefits, Asia-Africa-Europe trade. "
        "Risk: Describing only one port without the regional context. "
        "Distinct from EL-TRAIN-001: Logistics focus, not energy."
    ),
    prompt="لخّص أهمية الموانئ الخليجية الكبرى في سلاسل الإمداد الإقليمية والدولية.",
    response=(
        "تُشكّل الموانئ الخليجية الكبرى عقداً محورية في سلاسل الإمداد الإقليمية والدولية. "
        "يتصدّر ميناء جبل علي في دبي هذه الموانئ بوصفه أكبر ميناء في الشرق الأوسط وأفريقيا. "
        "تُوفر هذه الموانئ ثلاث وظائف رئيسية: "
        "أولاً: خدمات اللوجستيات المتكاملة بما تشمل التخزين والتوزيع وإعادة التصدير. "
        "ثانياً: المناطق الحرة التي تُقدّم إعفاءات جمركية وضريبية لجذب الاستثمار. "
        "ثالثاً: ربط الممرات التجارية بين آسيا وأفريقيا وأوروبا. "
        "تستند هذه المعلومات إلى بيانات متاحة للعموم."
    ),
    factual_dep=True,
))

TRAIN_EXAMPLES.append(make_example(
    family="EL-TRAIN-005", lang="ar", domain="energy_logistics",
    task_type="risk_identification", difficulty="intermediate",
    country="Kuwait", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify risks for a fictional Kuwaiti logistics company expanding into cold storage. "
        "Core facts: Fictional company in a hot climate with high energy costs. "
        "Required reasoning: Identify four risks specific to cold storage in Kuwait. "
        "Expected knowledge: Energy costs, food safety compliance, power outage risk, competition. "
        "Risk: Listing generic logistics risks without the cold storage specifics. "
        "Distinct from EL-TRAIN-003: Cold storage logistics, not petrochemical exports."
    ),
    prompt="شركة لوجستية كويتية افتراضية تدرس التوسع في خدمات التخزين البارد. ما المخاطر الرئيسية؟",
    response=(
        "تشمل المخاطر الرئيسية لشركة لوجستية كويتية افتراضية تتوسع في التخزين البارد: "
        "أولاً: ارتفاع تكاليف الطاقة — تستهلك مرافق التبريد كميات كبيرة من الكهرباء في المناخ الخليجي الحار، "
        "مما يُشكّل عبئاً تشغيلياً ثابتاً يُؤثر على الهوامش. "
        "ثانياً: متطلبات الامتثال الغذائي — تخضع مرافق التخزين البارد لرقابة صارمة من الجهات الصحية "
        "والبلدية، وأي إخفاق في الامتثال قد يُفضي إلى إغلاق المنشأة. "
        "ثالثاً: مخاطر انقطاع التيار الكهربائي — أي انقطاع غير مخطط له قد يُتلف البضائع المخزنة "
        "ويُعرّض الشركة لمطالبات تعويض. "
        "رابعاً: التنافسية — دخول مشغلين دوليين كبار قد يُضغط على الأسعار ويُقلّص الحصة السوقية. "
        "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="EL-TRAIN-006", lang="en", domain="energy_logistics",
    task_type="hallucination_resistance", difficulty="basic",
    country="GCC", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to provide Kuwait's current OPEC+ production quota. "
        "Core facts: OPEC+ quotas change at ministerial meetings and are not available in real-time. "
        "Required reasoning: Direct the user to OPEC's official website. "
        "Expected knowledge: OPEC website URL; KPC as the relevant national entity. "
        "Risk: Inventing a production quota figure. "
        "Distinct from GB-TRAIN-005: Energy production data, not banking rates."
    ),
    prompt="What is the current oil production quota assigned to Kuwait by OPEC+?",
    response=(
        "I do not have access to real-time OPEC+ production quota data. Quotas are subject to "
        "frequent revision at OPEC+ ministerial meetings and change based on market conditions. "
        "For the most current and authoritative figures, please consult the official OPEC "
        "website at www.opec.org or Kuwait Petroleum Corporation's official communications. "
        "Providing an invented quota figure could mislead energy market analysis."
    ),
))

# ─── Executive Decision — Training ────────────────────────────────────────

TRAIN_EXAMPLES.append(make_example(
    family="ED-TRAIN-001", lang="en", domain="executive_decision",
    task_type="decision_comparison", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Compare organic growth versus acquisition for a fictional GCC financial services firm. "
        "Core facts: Fictional firm with capital to deploy in a competitive market. "
        "Required reasoning: Identify the trade-offs in speed, risk, and cultural fit. "
        "Expected knowledge: Integration risk, valuation risk, cultural alignment, capital efficiency. "
        "Risk: Recommending one option without acknowledging the context-dependence. "
        "Distinct from others: Strategic decision comparison, not regulatory or technical."
    ),
    prompt="Compare organic growth versus acquisition as growth strategies for a fictional GCC financial services firm with available capital.",
    response=(
        "Organic growth preserves cultural coherence and avoids integration risk, but is slower "
        "and may not achieve the scale required to compete with established regional players. "
        "Acquisition accelerates market entry and capability acquisition but introduces "
        "integration complexity, cultural misalignment risk, and valuation risk — particularly "
        "in GCC markets where quality targets are scarce and pricing can be elevated. "
        "Key considerations for the fictional firm: "
        "(1) If the strategic objective is speed to market, acquisition is preferable provided "
        "rigorous due diligence is conducted and synergies are quantified before signing. "
        "(2) If cultural fit and brand consistency are priorities, organic growth with selective "
        "talent acquisition may be more appropriate. "
        "(3) A hybrid approach — organic growth supplemented by acqui-hires of specialist teams — "
        "can balance speed and cultural risk. "
        "The optimal path depends on the firm's capital position, risk appetite, and timeline. "
        "Note: This analysis uses a fictional firm for training purposes."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="ED-TRAIN-002", lang="ar", domain="executive_decision",
    task_type="executive_briefing", difficulty="intermediate",
    country="Kuwait", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Brief on digital transformation considerations for Kuwaiti government institutions. "
        "Core facts: Kuwaiti government is pursuing digital transformation. "
        "Required reasoning: Identify three key challenges: change management, talent, integration. "
        "Expected knowledge: Legacy systems, digital skills gap, citizen experience. "
        "Risk: Describing digital transformation as only a technology problem. "
        "Distinct from ED-TRAIN-001: Government sector, not financial services."
    ),
    prompt="أعدّ موجزاً تنفيذياً حول اعتبارات التحول الرقمي للمؤسسات الحكومية الكويتية.",
    response=(
        "يُمثّل التحول الرقمي للمؤسسات الحكومية الكويتية أولوية استراتيجية تتضمن رقمنة الخدمات الحكومية "
        "وتحديث البنية التحتية لتقنية المعلومات وتعزيز قدرات الأمن السيبراني. "
        "تبرز ثلاثة تحديات رئيسية: "
        "أولاً: إدارة التغيير المؤسسي — يتطلب التحول الرقمي تغييراً في الثقافة التنظيمية وليس مجرد تحديث تقني. "
        "ثانياً: توافر الكوادر البشرية المؤهلة — يُشكّل نقص المتخصصين في التحليل الرقمي وإدارة البيانات عائقاً رئيسياً. "
        "ثالثاً: ضمان التكامل بين الأنظمة القائمة والحلول الجديدة — تُعاني كثير من المؤسسات من أنظمة موروثة يصعب دمجها. "
        "ملاحظة: هذا تحليل افتراضي لأغراض التدريب."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="ED-TRAIN-003", lang="en", domain="executive_decision",
    task_type="risk_identification", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify top five strategic risks for fictional GCC family businesses. "
        "Core facts: Fictional family businesses facing succession and digital disruption. "
        "Required reasoning: Identify five risks specific to family business governance. "
        "Expected knowledge: Succession, digital disruption, talent, capital allocation, regulatory. "
        "Risk: Listing generic business risks without the family business governance context. "
        "Distinct from GB-TRAIN-003: Family business governance risks, not credit risks."
    ),
    prompt="What are the top five strategic risks facing fictional GCC family businesses in the next decade?",
    response=(
        "Top five strategic risks for fictional GCC family businesses: "
        "(1) Succession and governance: Transitioning leadership without formalised governance "
        "structures — such as a family council, shareholder agreement, or independent board — "
        "creates conflict risk and strategic discontinuity. "
        "(2) Digital disruption: Competition from digitally native entrants in retail, real "
        "estate, and services can erode market share faster than traditional businesses can adapt. "
        "(3) Talent retention: Attracting and retaining non-family professional management is "
        "increasingly challenging as compensation expectations rise. "
        "(4) Capital allocation discipline: Without institutional oversight, capital may be "
        "deployed in related-party transactions or non-core investments suboptimally. "
        "(5) Regulatory change: Evolving corporate governance requirements in GCC jurisdictions "
        "may require structural adjustments to ownership and reporting. "
        "Note: This analysis uses fictional businesses for training purposes."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="ED-TRAIN-004", lang="ar", domain="executive_decision",
    task_type="objection_handling", difficulty="intermediate",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Respond to a board member's objection that AI investment is too expensive. "
        "Core facts: Board member is concerned about ROI of AI investment. "
        "Required reasoning: Provide four counter-arguments addressing cost, ROI, and competitive risk. "
        "Expected knowledge: Cost of inaction, pilot approach, pre-built solutions vs training. "
        "Risk: Dismissing the objection without acknowledging legitimate cost concerns. "
        "Distinct from IF-TRAIN-022: AI investment objection, not Islamic finance objection."
    ),
    prompt="كيف يمكن الرد على اعتراض أحد أعضاء مجلس الإدارة بأن الاستثمار في الذكاء الاصطناعي مكلف للغاية؟",
    response=(
        "يمكن معالجة هذا الاعتراض المشروع من خلال أربعة محاور: "
        "أولاً: التمييز بين تكاليف تدريب النماذج الكبرى وتكاليف تبني حلول الذكاء الاصطناعي الجاهزة "
        "الأكثر اقتصادية والمتاحة بنماذج اشتراك. "
        "ثانياً: تقديم حالات استخدام محددة مع تقدير عائد الاستثمار المتوقع وفق معطيات الشركة الفعلية. "
        "ثالثاً: اقتراح نهج تدريجي يبدأ بمشاريع تجريبية محدودة التكلفة لإثبات القيمة قبل التوسع. "
        "رابعاً: الإشارة إلى تكلفة عدم الاستثمار في ظل تنامي اعتماد المنافسين على هذه التقنيات، "
        "مما قد يُؤدي إلى فجوة تنافسية يصعب تضييقها لاحقاً. "
        "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="ED-TRAIN-005", lang="en", domain="executive_decision",
    task_type="request_for_clarification", difficulty="basic",
    country="GCC", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Request clarification before advising on GCC market investment. "
        "Core facts: The question lacks context about industry, country, and investment type. "
        "Required reasoning: Identify five pieces of information needed for a meaningful analysis. "
        "Expected knowledge: Industry, country, horizon, entry mode, strategic objective. "
        "Risk: Providing generic advice without understanding the specific context. "
        "Distinct from GR-TRAIN-007: Investment clarification, not regulatory penalty."
    ),
    prompt="Should our company invest in the GCC market?",
    response=(
        "To provide a meaningful analysis, I need additional context: "
        "(1) What is your company's industry and current geographic footprint? "
        "(2) Which specific GCC country or countries are you considering? "
        "(3) What is the investment horizon and capital available? "
        "(4) Are you considering organic entry, partnership, or acquisition? "
        "(5) What is the primary strategic objective — revenue growth, cost optimisation, "
        "or geographic diversification? "
        "With these details, I can offer a more targeted and useful assessment."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="ED-TRAIN-006", lang="ar-en", domain="executive_decision",
    task_type="structured_extraction", difficulty="intermediate",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Extract key decision criteria from a fictional executive summary with three options. "
        "Core facts: Three fictional strategic options with different investment levels and ROI. "
        "Required reasoning: Tabulate the key criteria in both Arabic and English. "
        "Expected knowledge: Investment, ROI, risk, timeline. "
        "Risk: Omitting the risk column which is the key differentiator. "
        "Distinct from IF-TRAIN-024: Strategic decision, not Sukuk terms."
    ),
    prompt=(
        "Extract the key decision criteria from this fictional executive summary:\n"
        "Option A: Domestic expansion, investment required, ROI projected over 5 years, lower risk. "
        "Option B: Saudi joint venture, lower investment, higher ROI projected, higher execution risk. "
        "Option C: Defer decision pending regulatory clarity, no investment, opportunity cost only."
    ),
    response=(
        "| الخيار / Option | الاستثمار / Investment | العائد المتوقع / Projected ROI | المخاطر / Risk |\n"
        "|---|---|---|---|\n"
        "| A — التوسع المحلي / Domestic expansion | مرتفع / High | معتدل / Moderate | منخفض / Lower |\n"
        "| B — شراكة سعودية / Saudi JV | معتدل / Moderate | مرتفع / Higher | تنفيذ وشريك / Execution + partner |\n"
        "| C — التأجيل / Defer | لا يوجد / None | لا ينطبق / N/A | تكلفة الفرصة / Opportunity cost |\n\n"
        "Note: All options and figures are fictional and for training purposes only."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="ED-TRAIN-007", lang="en", domain="executive_decision",
    task_type="hallucination_resistance", difficulty="intermediate",
    country="Kuwait", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to provide Kuwait's exact GDP growth rate for a specific year. "
        "Core facts: GDP statistics are revised after initial publication. "
        "Required reasoning: Direct the user to authoritative statistical sources. "
        "Expected knowledge: CSB Kuwait, IMF, World Bank as authoritative sources. "
        "Risk: Inventing a GDP figure that could mislead economic analysis. "
        "Distinct from GB-TRAIN-005: Macroeconomic data, not banking rates."
    ),
    prompt="What was the exact GDP growth rate of Kuwait in 2024?",
    response=(
        "I do not have access to verified final GDP figures for Kuwait for 2024. "
        "Economic statistics are typically revised after initial publication and final figures "
        "may differ from preliminary estimates. "
        "For authoritative figures, please consult the Central Statistical Bureau of Kuwait, "
        "the IMF's World Economic Outlook database, or the World Bank's data portal. "
        "Providing an invented GDP figure could mislead economic or investment analysis."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="ED-TRAIN-008", lang="ar", domain="executive_decision",
    task_type="question_answering", difficulty="advanced",
    country="Kuwait", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify key valuation considerations for a fictional Kuwaiti investment firm acquiring a tech startup. "
        "Core facts: Fictional investment firm with capital looking at a tech startup. "
        "Required reasoning: Identify four valuation challenges specific to early-stage tech. "
        "Expected knowledge: No stable earnings, team retention, regulatory approval, tech fit. "
        "Risk: Applying standard DCF without acknowledging the limitations for early-stage tech. "
        "Distinct from ED-TRAIN-001: Acquisition valuation, not organic vs acquisition strategy."
    ),
    prompt="شركة استثمارية كويتية افتراضية تدرس الاستحواذ على شركة تقنية ناشئة. ما الاعتبارات الرئيسية للتقييم؟",
    response=(
        "في تقييم استحواذ شركة استثمارية كويتية افتراضية على شركة تقنية ناشئة، تبرز أربعة اعتبارات رئيسية: "
        "أولاً: التقييم في غياب أرباح مستقرة — يتطلب نماذج تقييم متخصصة كمضاعفات الإيرادات أو قيمة المستخدم، "
        "لا نموذج التدفقات النقدية المخصومة التقليدي. "
        "ثانياً: الفجوة التقنية — تقييم مدى توافق التقنية مع استراتيجية المستحوذ وقدرته على دمجها. "
        "ثالثاً: الاحتفاظ بالكوادر — ضمان بقاء الفريق المؤسس والكفاءات التقنية الرئيسية بعد الاستحواذ "
        "من خلال حوافز مناسبة. "
        "رابعاً: الإطار التنظيمي — التحقق من متطلبات الموافقات التنظيمية في الكويت وأي قيود على الملكية الأجنبية. "
        "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."
    ),
))

# ─── Arabic-English Correspondence — Training ─────────────────────────────

TRAIN_EXAMPLES.append(make_example(
    family="AC-TRAIN-001", lang="ar", domain="arabic_english_correspondence",
    task_type="formal_correspondence", difficulty="basic",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write a formal Arabic letter confirming attendance at a business meeting. "
        "Core facts: Standard business meeting confirmation. "
        "Required reasoning: Include date, time, location, and attendees. "
        "Expected knowledge: Formal Arabic letter structure and salutation. "
        "Risk: Using informal language in a formal business context. "
        "Distinct from IF-TRAIN-021: General business meeting, not Sukuk financing."
    ),
    prompt="اكتب رسالة رسمية باللغة العربية لتأكيد حضور اجتماع عمل مهم.",
    response=(
        "الموضوع: تأكيد حضور الاجتماع\n\nالسادة / [اسم الجهة]\nتحية طيبة وبعد،\n\n"
        "يسعدنا تأكيد حضورنا للاجتماع المقرر بتاريخ [التاريخ] في تمام الساعة [الوقت] في [المكان]. "
        "سيمثّل شركتنا في هذا الاجتماع كل من [الاسم الأول] و[الاسم الثاني]. "
        "نتطلع إلى لقاء فريقكم ومناقشة المحاور المدرجة في جدول الأعمال.\n\n"
        "مع خالص التقدير والاحترام،\n[الاسم] — [المسمى الوظيفي]"
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="AC-TRAIN-002", lang="en", domain="arabic_english_correspondence",
    task_type="formal_correspondence", difficulty="basic",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write a formal English email declining a business proposal politely. "
        "Core facts: Standard business proposal rejection. "
        "Required reasoning: Decline without burning the relationship; keep the door open. "
        "Expected knowledge: Professional English email structure; diplomatic language. "
        "Risk: Being too blunt or too vague about the reason for declining. "
        "Distinct from AC-TRAIN-001: English declination, not Arabic confirmation."
    ),
    prompt="Write a formal English email politely declining a business proposal.",
    response=(
        "Subject: RE: Business Proposal\n\nDear [Name],\n\n"
        "Thank you for presenting your proposal and for the time you invested in preparing it. "
        "After careful consideration, we have decided not to proceed with this opportunity at this time. "
        "This decision reflects our current strategic priorities rather than any deficiency in your proposal.\n\n"
        "We appreciate your interest in working with us and hope there may be opportunities "
        "to collaborate in the future.\n\n"
        "Kind regards,\n[Name] — [Title]"
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="AC-TRAIN-003", lang="ar-en", domain="arabic_english_correspondence",
    task_type="arabic_english_translation", difficulty="intermediate",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'We look forward to a mutually beneficial partnership' into formal Arabic. "
        "Core facts: Standard business closing phrase requiring formal Arabic register. "
        "Required reasoning: Provide the standard translation and a more formal GCC variant. "
        "Expected knowledge: GCC formal Arabic register; honorific language conventions. "
        "Risk: Using informal Arabic that sounds unprofessional in GCC business context. "
        "Distinct from IF-TRAIN-009: Business correspondence phrase, not Islamic finance term."
    ),
    prompt="Translate the business phrase 'We look forward to a mutually beneficial partnership' into formal Arabic suitable for GCC business correspondence.",
    response=(
        "الترجمة الرسمية: «نتطلع إلى شراكة مثمرة تعود بالنفع على الطرفين.»\n\n"
        "في السياقات الرسمية الخليجية، يمكن استخدام صيغة أكثر رسمية وتوقيراً: "
        "«نأمل أن تكون هذه الشراكة نموذجاً للتعاون المثمر الذي يُحقق المصالح المشتركة لكلا الطرفين.»\n\n"
        "Note: The formal Arabic register uses honorific language that is standard in GCC "
        "business correspondence and conveys respect for the relationship."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="AC-TRAIN-004", lang="ar", domain="arabic_english_correspondence",
    task_type="document_summarization", difficulty="basic",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Summarize the key elements of a formal Arabic letter requesting a project deadline extension. "
        "Core facts: Standard request for timeline adjustment. "
        "Required reasoning: Identify the four essential elements: reference, dates, reason, plan. "
        "Expected knowledge: Formal Arabic letter structure; professional tone. "
        "Risk: Omitting the revised work plan which is essential for credibility. "
        "Distinct from AC-TRAIN-001: Deadline extension request, not meeting confirmation."
    ),
    prompt="لخّص العناصر الرئيسية التي يجب أن تتضمنها رسالة عمل رسمية تطلب تمديد موعد تسليم مشروع.",
    response=(
        "رسالة العمل الرسمية لطلب تمديد الموعد يجب أن تتضمن أربعة عناصر رئيسية: "
        "أولاً: الإشارة الواضحة إلى المشروع ورقم المرجع والموعد الأصلي للتسليم. "
        "ثانياً: الموعد الجديد المقترح مع الالتزام الصريح بالتسليم في هذا الموعد. "
        "ثالثاً: توضيح الأسباب الموضوعية للتأخير بصورة مختصرة وصادقة دون مبالغة. "
        "رابعاً: إرفاق خطة عمل معدّلة توضح الخطوات المتبقية والجدول الزمني المحدّث. "
        "يُستحسن تقديم الطلب مبكراً قبل الموعد الأصلي لإتاحة الوقت الكافي للمراجعة."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="AC-TRAIN-005", lang="ar-en", domain="arabic_english_correspondence",
    task_type="formal_correspondence", difficulty="intermediate",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write a bilingual meeting invitation subject line and opening paragraph. "
        "Core facts: Strategic collaboration meeting requiring bilingual communication. "
        "Required reasoning: Provide parallel Arabic and English text for the same content. "
        "Expected knowledge: Bilingual business communication; parallel structure. "
        "Risk: Using different content in the Arabic and English versions. "
        "Distinct from AC-TRAIN-001: Bilingual format, not monolingual Arabic."
    ),
    prompt="Write a bilingual (Arabic and English) subject line and opening paragraph for a strategic collaboration meeting invitation.",
    response=(
        "الموضوع / Subject:\n"
        "AR: دعوة لحضور اجتماع: مناقشة فرص التعاون الاستراتيجي\n"
        "EN: Meeting Invitation: Strategic Collaboration Discussion\n\n"
        "الفقرة الافتتاحية / Opening Paragraph:\n"
        "AR: يسرّنا دعوتكم لحضور اجتماع لمناقشة فرص التعاون الاستراتيجي بين مؤسستينا، "
        "بتاريخ [التاريخ] في [المكان].\n"
        "EN: We are pleased to invite you to a meeting to discuss strategic collaboration "
        "opportunities between our organisations, scheduled for [Date] at [Location]."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="AC-TRAIN-006", lang="en", domain="arabic_english_correspondence",
    task_type="objection_handling", difficulty="intermediate",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write a formal response to a client complaint about service delays. "
        "Core facts: Client has complained about a service delay. "
        "Required reasoning: Acknowledge, apologise, explain, remedy, recommit. "
        "Expected knowledge: Professional complaint response structure; empathetic tone. "
        "Risk: Being defensive or providing too much technical detail. "
        "Distinct from AC-TRAIN-002: Complaint response, not proposal declination."
    ),
    prompt="How should a GCC executive respond in writing to a client complaint about service delays?",
    response=(
        "An effective written response to a service delay complaint should follow five steps: "
        "(1) Acknowledge the delay promptly and without deflection or minimisation. "
        "(2) Express a sincere apology for the inconvenience caused to the client. "
        "(3) Provide a factual, concise explanation of the root cause without excessive technical detail. "
        "(4) State the specific corrective actions taken or planned, with realistic timelines. "
        "(5) Offer a concrete remedy if appropriate, and reaffirm the commitment to service quality. "
        "The tone should be professional, empathetic, and solution-focused throughout. "
        "Note: This guidance uses a fictional scenario for training purposes."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="AC-TRAIN-007", lang="ar", domain="arabic_english_correspondence",
    task_type="refusal_missing_info", difficulty="basic",
    country="GCC", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to write a formal letter without knowing the subject or recipient. "
        "Core facts: The request lacks essential information for drafting a meaningful letter. "
        "Required reasoning: Identify the four pieces of information needed. "
        "Expected knowledge: Purpose, recipient, topic, background. "
        "Risk: Writing a generic placeholder letter that is useless. "
        "Distinct from ED-TRAIN-005: Correspondence task, not investment advice."
    ),
    prompt="هل يمكنك كتابة رسالة رسمية نيابةً عني دون أن تعرف تفاصيل الموضوع أو الجهة المُرسَل إليها؟",
    response=(
        "لا يمكنني كتابة رسالة رسمية فعّالة دون معرفة التفاصيل الأساسية. أحتاج على الأقل إلى: "
        "أولاً: الغرض من الرسالة والهدف المطلوب تحقيقه. "
        "ثانياً: الجهة المُرسَل إليها ومسمى المسؤول المعني وعنوانه. "
        "ثالثاً: الموضوع الرئيسي والنقاط التي تودّ إيصالها. "
        "رابعاً: أي معلومات خلفية أو مرفقات ذات صلة تُعزز الرسالة. "
        "يُرجى تزويدي بهذه المعلومات لأتمكن من صياغة رسالة مناسبة وملائمة للسياق."
    ),
))

TRAIN_EXAMPLES.append(make_example(
    family="AC-TRAIN-008", lang="ar-en", domain="arabic_english_correspondence",
    task_type="arabic_english_translation", difficulty="advanced",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate a formal Arabic business phrase into English preserving the formal register. "
        "Core facts: The Arabic phrase uses highly formal honorific language. "
        "Required reasoning: Preserve the formal register while adapting to English conventions. "
        "Expected knowledge: Difference between Arabic and English formal register conventions. "
        "Risk: Translating too literally, producing unnatural English. "
        "Distinct from AC-TRAIN-003: Translating from Arabic to English, not English to Arabic."
    ),
    prompt=(
        "Translate this formal Arabic business phrase into English, preserving the formal register:\n"
        "«نتشرف بتقديم عروضنا لخدماتكم الكريمة، ونأمل أن تجدوا فيها ما يلبّي تطلعاتكم.»"
    ),
    response=(
        "Translation: 'We are honoured to present our services for your esteemed consideration, "
        "and we hope you will find in them what meets your aspirations.'\n\n"
        "Note on register: The Arabic phrase uses highly formal honorific language common in GCC "
        "business correspondence, including the honorific 'الكريمة' (esteemed) and the deferential "
        "'نتشرف' (we are honoured). The English translation preserves the formal tone while adapting "
        "to standard English business letter conventions, which are somewhat less elaborate."
    ),
))

# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK LIBRARY — examples independently authored for VALID and EVAL
# These are NOT derived from training examples.
# Each has an independent scenario brief, prompt, and response.
# ═══════════════════════════════════════════════════════════════════════════

BENCH_EXAMPLES: List[Dict] = []

# ─── Islamic Finance — Benchmark ──────────────────────────────────────────

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-001", lang="ar", domain="islamic_finance",
    task_type="question_answering", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Explain the Waqf (Islamic endowment) and its role in social finance. "
        "Core facts: Waqf is a perpetual endowment for charitable purposes. "
        "Required reasoning: Explain the three types of Waqf and their governance. "
        "Expected knowledge: Waqf al-dhurri, Waqf al-khayri, Waqf al-mushtarak. "
        "Risk: Confusing Waqf with Zakat. "
        "Distinct from all training examples: Covers Waqf, not covered in training set."
    ),
    prompt="ما الوقف الإسلامي وما أنواعه الرئيسية ودوره في التمويل الاجتماعي؟",
    response=(
        "الوقف الإسلامي هو تخصيص أصل بصفة دائمة لأغراض خيرية أو اجتماعية، "
        "بحيث لا يُباع الأصل ولا يُوهب ولا يُورَّث، ويُستفاد من ريعه فقط. "
        "ينقسم الوقف إلى ثلاثة أنواع: "
        "أولاً: الوقف الذري — يُخصَّص ريعه لذرية الواقف أو أقاربه. "
        "ثانياً: الوقف الخيري — يُخصَّص ريعه لأغراض خيرية عامة كالمساجد والمستشفيات والمدارس. "
        "ثالثاً: الوقف المشترك — يجمع بين الذري والخيري. "
        "يُسهم الوقف في التمويل الاجتماعي من خلال توليد دخل مستدام للمشاريع الخيرية "
        "دون الاعتماد على التبرعات المتكررة."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-002", lang="en", domain="islamic_finance",
    task_type="regulatory_explanation", difficulty="advanced",
    country="GCC", prov_type="SOURCE_DEPENDENT_FACTUAL",
    source_name="IFSB Islamic Financial Services Industry Stability Report",
    source_url="https://www.ifsb.org/stability-report/",
    scenario_brief=(
        "Task: Explain the IFSB's approach to liquidity risk management for Islamic banks. "
        "Core facts: IFSB-12 addresses liquidity risk for Islamic financial institutions. "
        "Required reasoning: Explain why conventional liquidity tools are not directly applicable. "
        "Expected knowledge: Prohibition on interest-bearing instruments, Sharia-compliant liquidity tools. "
        "Risk: Applying conventional LCR/NSFR without Islamic adjustments. "
        "Distinct from all training examples: Liquidity risk management, not covered in training."
    ),
    prompt="How does the IFSB approach liquidity risk management for Islamic banks, and why do conventional tools require adaptation?",
    response=(
        "The IFSB has issued IFSB-12, the Guiding Principles on Liquidity Risk Management for "
        "Institutions Offering Islamic Financial Services. Conventional liquidity management tools "
        "require adaptation because Islamic banks cannot hold or issue interest-bearing instruments "
        "to manage short-term liquidity. "
        "Key adaptations include: "
        "(1) Sharia-compliant high-quality liquid assets (HQLA) such as sovereign sukuk and "
        "central bank commodity murabaha facilities replace conventional government bonds. "
        "(2) Interbank liquidity is managed through commodity murabaha or wakala structures "
        "rather than conventional overnight lending. "
        "(3) The liquidity coverage ratio (LCR) and net stable funding ratio (NSFR) are adapted "
        "to reflect the different maturity profiles of Islamic financing modes. "
        "Specific current requirements should be verified against the latest IFSB standards "
        "and applicable central bank regulations."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-003", lang="ar-en", domain="islamic_finance",
    task_type="arabic_english_translation", difficulty="intermediate",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'Diminishing Musharaka' into Arabic with a bilingual definition. "
        "Core facts: A specific variant of Musharaka used in home finance. "
        "Required reasoning: Explain the progressive equity acquisition mechanism. "
        "Expected knowledge: Co-ownership, rental payments, equity unit purchases. "
        "Risk: Describing it as simply 'decreasing partnership' without the equity mechanism. "
        "Distinct from IF-TRAIN-009/010/011: Different term with different mechanism."
    ),
    prompt="Translate 'Diminishing Musharaka' into Arabic and provide a bilingual definition explaining how equity transfers.",
    response=(
        "الترجمة: المشاركة المتناقصة.\n\n"
        "التعريف بالعربية: المشاركة المتناقصة هيكل تمويلي يشترك فيه البنك والعميل في ملكية الأصل، "
        "ويقوم العميل بشراء حصص البنك تدريجياً حتى يمتلك الأصل كاملاً.\n\n"
        "English definition: Diminishing Musharaka is a co-ownership structure in which the bank "
        "and client jointly own an asset. The client pays rent on the bank's share and "
        "simultaneously purchases equity units from the bank over time, progressively increasing "
        "ownership until full title is acquired."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-004", lang="ar", domain="islamic_finance",
    task_type="risk_identification", difficulty="advanced",
    country="Kuwait", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify Sharia risks in a hypothetical Mudaraba investment fund. "
        "Core facts: Fictional fund where bank acts as mudarib managing client capital. "
        "Required reasoning: Identify four Sharia risks specific to the mudarib role. "
        "Expected knowledge: Guaranteed return prohibition, commingling, disclosure, exit. "
        "Risk: Confusing Mudaraba risks with Musharaka risks. "
        "Distinct from IF-TRAIN-006: Mudaraba fund risks, not Ijarah structure risks."
    ),
    prompt="صندوق استثماري مضاربة افتراضي يعمل فيه البنك مضارباً. ما المخاطر الشرعية الرئيسية؟",
    response=(
        "تشمل المخاطر الشرعية الرئيسية في صندوق مضاربة افتراضي يعمل فيه البنك مضارباً: "
        "أولاً: ضمان العائد — لا يجوز للمضارب ضمان عائد محدد لأصحاب رأس المال، "
        "وأي التزام بعائد ثابت يُحوّل العقد إلى قرض بفائدة. "
        "ثانياً: خلط الأموال — يجب فصل أموال المضاربة عن أموال البنك الخاصة بصورة واضحة. "
        "ثالثاً: الإفصاح — يجب الإفصاح الكامل عن استراتيجية الاستثمار والرسوم والمخاطر لأصحاب رأس المال. "
        "رابعاً: آلية الخروج — يجب أن تكون آلية تصفية الصندوق وتوزيع الأصول محددة مسبقاً وموافقاً عليها شرعاً. "
        "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-005", lang="en", domain="islamic_finance",
    task_type="question_answering", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Explain the concept of Qard al-Hasan in Islamic banking. "
        "Core facts: Interest-free loan for social purposes. "
        "Required reasoning: Explain who provides it, who receives it, and what is repaid. "
        "Expected knowledge: No interest, principal only, social welfare purpose. "
        "Risk: Confusing Qard al-Hasan with a conventional zero-interest promotional loan. "
        "Distinct from all training examples: Covers Qard al-Hasan, not in training set."
    ),
    prompt="What is Qard al-Hasan in Islamic banking and how does it differ from a conventional zero-interest loan?",
    response=(
        "Qard al-Hasan is an interest-free loan provided as an act of benevolence, where only "
        "the principal amount is repaid without any addition. "
        "It differs from a conventional zero-interest promotional loan in several ways: "
        "First, Qard al-Hasan is motivated by religious and social welfare obligations, not "
        "commercial incentives. "
        "Second, the lender has no expectation of any financial benefit, direct or indirect. "
        "Third, it is typically offered to individuals in financial need or for social purposes "
        "such as education or healthcare. "
        "Some Islamic banks offer Qard al-Hasan programmes as part of their social responsibility "
        "mandate, funded from Zakat or charitable reserves."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-006", lang="ar", domain="islamic_finance",
    task_type="document_summarization", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Summarize the key principles of Takaful under AAOIFI standards. "
        "Core facts: AAOIFI has specific standards for Takaful operators. "
        "Required reasoning: Identify the three models: Wakala, Mudaraba, hybrid. "
        "Expected knowledge: Tabarru concept, surplus distribution, investment of funds. "
        "Risk: Describing only one Takaful model without the alternatives. "
        "Distinct from IF-TRAIN-011: Summary of AAOIFI principles, not a translation task."
    ),
    prompt="لخّص المبادئ الأساسية للتأمين التكافلي وفق معايير هيئة أيوفي.",
    response=(
        "تُحدّد معايير أيوفي للتأمين التكافلي ثلاثة مبادئ جوهرية: "
        "أولاً: التبرع (Tabarru) — يُودع كل مشترك جزءاً من قسطه في صندوق التبرع المشترك لتغطية الأضرار. "
        "ثانياً: نماذج إدارة الصندوق — تُجيز أيوفي ثلاثة نماذج: الوكالة حيث يتقاضى المشغّل رسوماً ثابتة، "
        "والمضاربة حيث يُشارك في الأرباح، والنموذج الهجين الجامع بينهما. "
        "ثالثاً: توزيع الفائض — يُوزَّع الفائض في صندوق التبرع على المشتركين وفق السياسة المعتمدة، "
        "ولا يجوز للمشغّل الاستئثار به. "
        "يُنصح بمراجعة المعيار الكامل والهيئة الشرعية للتطبيق الفعلي."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-007", lang="ar-en", domain="islamic_finance",
    task_type="structured_extraction", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Extract key terms from a fictional Mudaraba fund prospectus. "
        "Core facts: Fictional fund with specific size, horizon, and target return. "
        "Required reasoning: Identify and tabulate the key risk factors. "
        "Expected knowledge: Market risk, Sharia risk, manager risk, capital loss, liquidity. "
        "Risk: Omitting the capital loss risk which is specific to Mudaraba. "
        "Distinct from IF-TRAIN-024: Mudaraba fund, not Ijarah Sukuk."
    ),
    prompt=(
        "Extract the key risk factors from this fictional Mudaraba fund prospectus:\n"
        "Fund: Gulf Mudaraba Growth Fund (fictional). Manager: Al Noor AM (fictional). "
        "Capital: USD 100M. Horizon: 5 years. Target return: 8-12% per annum."
    ),
    response=(
        "| عامل المخاطرة / Risk Factor | الوصف / Description |\n|---|---|\n"
        "| تقلبات السوق / Market volatility | العوائد تعتمد على أداء الأصول الأساسية ولا يوجد ضمان |\n"
        "| مخاطر الامتثال الشرعي / Sharia compliance | قد تُطعن في الهيكل تفسيرات فقهية مختلفة |\n"
        "| أداء المضارب / Manager performance | العوائد تعتمد على مهارة وحكم المضارب |\n"
        "| خسارة رأس المال / Capital loss | الخسارة المالية تقع على رب المال في المضاربة |\n"
        "| مخاطر السيولة / Liquidity risk | أجل 5 سنوات يُقيّد خيارات الخروج المبكر |\n\n"
        "Note: All entities are fictional and for training purposes only."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-008", lang="en", domain="islamic_finance",
    task_type="hallucination_resistance", difficulty="basic",
    country="GCC", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to confirm whether a specific investment is halal without details. "
        "Core facts: Halal status depends on the specific investment structure and activities. "
        "Required reasoning: Explain what information is needed for a proper assessment. "
        "Expected knowledge: Screening criteria, Sharia board role, prohibited activities. "
        "Risk: Confirming halal status without reviewing the investment. "
        "Distinct from IF-TRAIN-013: English refusal about investment, not Arabic product compliance."
    ),
    prompt="Is investing in the XYZ Technology Fund halal?",
    response=(
        "I cannot confirm whether any specific investment is halal without reviewing its details. "
        "Halal status depends on: "
        "(1) The fund's investment screening criteria and prohibited activities list; "
        "(2) Whether the fund has obtained a Sharia board certification; "
        "(3) The specific companies or assets held and their primary business activities; and "
        "(4) The fund's debt and interest income thresholds. "
        "Please request the fund's Sharia compliance certificate and screening methodology "
        "directly from the fund manager or its Sharia board."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-009", lang="ar", domain="islamic_finance",
    task_type="question_answering", difficulty="advanced",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="AAOIFI", source_url="https://aaoifi.com/",
    scenario_brief=(
        "Task: Explain the Sharia requirements for a Salam contract. "
        "Core facts: Salam is a forward sale with advance payment. "
        "Required reasoning: Explain why advance payment is required and what can be sold. "
        "Expected knowledge: Full advance payment, fungible goods, specified delivery. "
        "Risk: Confusing Salam with Istisna (which allows deferred payment). "
        "Distinct from IF-TRAIN-005: Salam, not Istisna; different payment requirement."
    ),
    prompt="ما عقد السلم في الفقه الإسلامي وما المتطلبات الشرعية الرئيسية لصحته؟",
    response=(
        "السلم عقد بيع آجل يُدفع فيه الثمن كاملاً مقدماً مقابل سلعة موصوفة تُسلَّم في المستقبل. "
        "تشترط الشريعة الإسلامية لصحة عقد السلم: "
        "أولاً: دفع رأس مال السلم كاملاً في مجلس العقد دون تأجيل. "
        "ثانياً: أن تكون السلعة المبيعة مثلية قابلة للوصف الدقيق كالحبوب والمعادن. "
        "ثالثاً: تحديد الأجل والمكان والكمية والجودة بدقة تامة. "
        "رابعاً: لا يجوز بيع السلعة قبل قبضها إلا في السلم الموازي. "
        "يختلف السلم عن الاستصناع في أن السلم يشترط دفع الثمن كاملاً مقدماً."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="IF-BENCH-010", lang="en", domain="islamic_finance",
    task_type="executive_briefing", difficulty="intermediate",
    country="GCC", prov_type="SOURCE_DEPENDENT_FACTUAL",
    source_name="IFSB Islamic Financial Services Industry Stability Report",
    source_url="https://www.ifsb.org/stability-report/",
    scenario_brief=(
        "Task: Brief on the global Sukuk market and its growth drivers. "
        "Core facts: IFSB tracks global Sukuk issuance annually. "
        "Required reasoning: Identify the three main growth drivers. "
        "Expected knowledge: Government financing needs, infrastructure, ESG Sukuk. "
        "Risk: Citing specific issuance volumes that may be outdated. "
        "Distinct from IF-TRAIN-015: Sukuk market briefing, not general Islamic finance industry."
    ),
    prompt="Provide a brief executive briefing on the global Sukuk market and its key growth drivers.",
    response=(
        "The global Sukuk market has grown significantly over the past decade, driven by "
        "three primary factors: "
        "(1) Government financing needs: GCC and Southeast Asian governments have increasingly "
        "used Sukuk to finance budget deficits and infrastructure projects in a Sharia-compliant "
        "manner, broadening the investor base. "
        "(2) Infrastructure financing: Large-scale infrastructure projects in the GCC, Malaysia, "
        "and Turkey have used Sukuk structures to access Islamic capital markets. "
        "(3) ESG and green Sukuk: The alignment between Islamic finance principles and "
        "environmental, social, and governance (ESG) objectives has driven issuance of "
        "green and sustainability Sukuk. "
        "Accurate current issuance volumes should be verified against the latest IFSB "
        "Islamic Financial Services Industry Stability Report."
    ),
    factual_dep=True,
))

# ─── GCC Banking — Benchmark ──────────────────────────────────────────────

BENCH_EXAMPLES.append(make_example(
    family="GB-BENCH-001", lang="en", domain="gcc_banking",
    task_type="regulatory_explanation", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Basel Committee on Banking Supervision", source_url="https://www.bis.org/bcbs/",
    scenario_brief=(
        "Task: Explain the Net Stable Funding Ratio (NSFR) under Basel III. "
        "Core facts: NSFR measures long-term liquidity resilience. "
        "Required reasoning: Explain the ASF/RSF framework and the 100% minimum. "
        "Expected knowledge: Available stable funding, required stable funding, one-year horizon. "
        "Risk: Confusing NSFR with LCR (which measures short-term liquidity). "
        "Distinct from GB-TRAIN-001: NSFR liquidity metric, not capital adequacy."
    ),
    prompt="Explain the Net Stable Funding Ratio (NSFR) under Basel III and why it differs from the Liquidity Coverage Ratio.",
    response=(
        "The Net Stable Funding Ratio (NSFR) measures a bank's ability to fund its activities "
        "with stable sources of funding over a one-year horizon. "
        "The NSFR is calculated as Available Stable Funding (ASF) divided by Required Stable "
        "Funding (RSF), and must be at least 100%. "
        "ASF includes equity, long-term liabilities, and stable retail deposits. "
        "RSF includes illiquid assets, long-term loans, and off-balance-sheet commitments. "
        "The NSFR differs from the Liquidity Coverage Ratio (LCR) in time horizon: the LCR "
        "measures the ability to survive a 30-day stress scenario using high-quality liquid "
        "assets, while the NSFR addresses structural funding mismatches over one year. "
        "Both ratios are part of the Basel III liquidity framework."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="GB-BENCH-002", lang="ar", domain="gcc_banking",
    task_type="document_summarization", difficulty="intermediate",
    country="Kuwait", prov_type="SOURCE_DEPENDENT_FACTUAL",
    source_name="Central Bank of Kuwait Annual Report",
    source_url="https://www.cbk.gov.kw/en/statistics-and-publications/publications/annual-reports",
    scenario_brief=(
        "Task: Summarize the key financial stability indicators for Kuwait's banking sector. "
        "Core facts: CBK publishes annual financial stability indicators. "
        "Required reasoning: Identify the four main indicator categories. "
        "Expected knowledge: Capital adequacy, asset quality, liquidity, profitability. "
        "Risk: Citing specific figures that change annually. "
        "Distinct from GB-TRAIN-001: Stability indicators, not capital adequacy requirements."
    ),
    prompt="وفق التقارير السنوية لبنك الكويت المركزي، ما المؤشرات الرئيسية لسلامة القطاع المصرفي الكويتي؟",
    response=(
        "يُصدر بنك الكويت المركزي تقارير دورية تتضمن أربع فئات رئيسية من مؤشرات السلامة المالية: "
        "أولاً: مؤشرات كفاية رأس المال — تشمل نسبة رأس المال الإجمالي إلى الأصول المرجحة بالمخاطر. "
        "ثانياً: مؤشرات جودة الأصول — تشمل نسبة القروض المتعثرة إلى إجمالي القروض ومعدلات التغطية. "
        "ثالثاً: مؤشرات السيولة — تشمل نسبة الأصول السائلة إلى إجمالي الأصول. "
        "رابعاً: مؤشرات الربحية — تشمل العائد على الأصول والعائد على حقوق الملكية. "
        "تتغير هذه المؤشرات سنوياً ويجب التحقق منها من المصدر الرسمي على موقع البنك المركزي."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="GB-BENCH-003", lang="ar-en", domain="gcc_banking",
    task_type="formal_correspondence", difficulty="intermediate",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Draft a bilingual executive summary for a bank's annual results announcement. "
        "Core facts: Bank needs to communicate results in both Arabic and English. "
        "Required reasoning: Provide parallel Arabic and English content. "
        "Expected knowledge: Bilingual financial communication; key metrics to include. "
        "Risk: Using different content in the two language versions. "
        "Distinct from GB-TRAIN-007: Annual results announcement, not credit facility request."
    ),
    prompt="Draft a short bilingual (Arabic and English) executive summary template for a bank's annual financial results announcement.",
    response=(
        "AR: ملخص تنفيذي — النتائج المالية السنوية\n"
        "حقق [اسم البنك] خلال العام المالي المنتهي في [التاريخ] نتائج إيجابية، "
        "إذ بلغ صافي الربح [المبلغ]، وارتفعت نسبة كفاية رأس المال إلى [النسبة]%، "
        "مع الحفاظ على نسبة قروض متعثرة منخفضة بلغت [النسبة]%.\n\n"
        "EN: Executive Summary — Annual Financial Results\n"
        "[Bank Name] delivered positive results for the financial year ended [Date], "
        "with net profit of [Amount], a capital adequacy ratio of [Rate]%, "
        "and a non-performing loan ratio of [Rate]%.\n\n"
        "Note: Bracketed fields are placeholders; this is a training template only."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="GB-BENCH-004", lang="en", domain="gcc_banking",
    task_type="risk_identification", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify operational risks in a fictional GCC bank's digital transformation. "
        "Core facts: Fictional bank migrating core banking system to cloud. "
        "Required reasoning: Identify five operational risks specific to core banking migration. "
        "Expected knowledge: Data migration, system downtime, cybersecurity, regulatory approval, vendor lock-in. "
        "Risk: Listing generic IT risks without the banking-specific context. "
        "Distinct from GB-TRAIN-003: Operational risks in digital transformation, not credit risks."
    ),
    prompt="Identify the key operational risks in a fictional GCC bank's migration of its core banking system to a cloud platform.",
    response=(
        "Key operational risks in a fictional GCC bank's core banking cloud migration: "
        "(1) Data migration integrity: Errors in migrating customer account data, transaction "
        "history, or collateral records could cause financial loss and regulatory penalties. "
        "(2) System downtime: Extended unavailability during cutover could disrupt customer "
        "services and trigger regulatory scrutiny. "
        "(3) Cybersecurity: Cloud migration expands the attack surface; inadequate security "
        "controls could expose sensitive customer data. "
        "(4) Regulatory approval: GCC central banks require prior approval for significant "
        "technology changes; failure to obtain approval could result in enforcement action. "
        "(5) Vendor lock-in: Dependence on a single cloud provider creates concentration risk "
        "and may limit future flexibility. "
        "Note: This analysis uses a fictional bank for training purposes."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="GB-BENCH-005", lang="ar", domain="gcc_banking",
    task_type="question_answering", difficulty="basic",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="FATF", source_url="https://www.fatf-gafi.org/",
    scenario_brief=(
        "Task: Explain the concept of beneficial ownership in AML compliance. "
        "Core facts: FATF requires identification of beneficial owners. "
        "Required reasoning: Define beneficial ownership and the threshold for identification. "
        "Expected knowledge: 25% ownership threshold, control test, FATF Recommendation 24. "
        "Risk: Confusing legal ownership with beneficial ownership. "
        "Distinct from GB-TRAIN-004: Beneficial ownership concept, not general AML requirements."
    ),
    prompt="ما المقصود بالملكية الفعلية (Beneficial Ownership) في إطار مكافحة غسل الأموال وما أهميتها؟",
    response=(
        "الملكية الفعلية تُشير إلى الأشخاص الطبيعيين الذين يمتلكون أو يُسيطرون فعلياً على كيان قانوني، "
        "بصرف النظر عن الهيكل القانوني الرسمي للملكية. "
        "تُلزم توصيات مجموعة العمل المالي (FATF) المؤسسات المالية بالتحقق من هوية المالكين الفعليين "
        "الذين يمتلكون ما يزيد على 25% من حقوق الملكية أو التصويت في الشركة. "
        "تُعدّ الملكية الفعلية محورية في مكافحة غسل الأموال لأن المجرمين يستخدمون أحياناً هياكل "
        "قانونية معقدة لإخفاء هويتهم الحقيقية. "
        "يُنصح بمراجعة التعاميم الصادرة عن البنك المركزي المختص للاطلاع على المتطلبات التفصيلية."
    ),
    factual_dep=True,
))

# ─── Telecommunications — Benchmark ───────────────────────────────────────

BENCH_EXAMPLES.append(make_example(
    family="TC-BENCH-001", lang="en", domain="telecommunications",
    task_type="regulatory_explanation", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="ITU", source_url="https://www.itu.int/",
    scenario_brief=(
        "Task: Explain the concept of network neutrality and its regulatory status in the GCC. "
        "Core facts: Network neutrality is debated globally; GCC has specific approaches. "
        "Required reasoning: Define net neutrality and explain why it is relevant to GCC regulators. "
        "Expected knowledge: Traffic management, zero-rating, ISP obligations. "
        "Risk: Applying EU or US net neutrality rules to the GCC without qualification. "
        "Distinct from TC-TRAIN-001: Net neutrality concept, not MNO obligations."
    ),
    prompt="What is network neutrality and how do GCC telecom regulators approach it?",
    response=(
        "Network neutrality is the principle that internet service providers should treat all "
        "internet traffic equally, without discriminating based on source, destination, or "
        "content type. "
        "In practice, this means ISPs should not block, throttle, or prioritise specific "
        "applications or websites. "
        "GCC telecom regulators have taken varied approaches: some permit managed traffic "
        "practices for network efficiency, while others have introduced guidelines on "
        "zero-rating and traffic management. "
        "The specific regulatory position varies by country and is subject to change. "
        "For current requirements, please consult the relevant national regulator such as "
        "CITRA in Kuwait, TRA in the UAE, or CITC in Saudi Arabia."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="TC-BENCH-002", lang="ar", domain="telecommunications",
    task_type="question_answering", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="ITU", source_url="https://www.itu.int/",
    scenario_brief=(
        "Task: Explain the difference between MVNO and MNO in telecom regulation. "
        "Core facts: MVNOs use MNO infrastructure under wholesale agreements. "
        "Required reasoning: Explain the regulatory and commercial differences. "
        "Expected knowledge: Infrastructure ownership, spectrum licences, wholesale agreements. "
        "Risk: Treating MVNOs as having the same regulatory obligations as MNOs. "
        "Distinct from TC-TRAIN-002: MVNO vs MNO distinction, not QoS metrics."
    ),
    prompt="ما الفرق بين مشغل الشبكة المتنقلة الافتراضي (MVNO) ومشغل الشبكة المتنقلة (MNO) من الناحية التنظيمية؟",
    response=(
        "مشغل الشبكة المتنقلة (MNO) هو شركة تمتلك بنيتها التحتية الخاصة وترخيص الطيف الترددي، "
        "وتتحمل المسؤولية الكاملة عن تشغيل الشبكة وجودة الخدمة. "
        "أما مشغل الشبكة المتنقلة الافتراضي (MVNO) فهو شركة تقدم خدمات الاتصالات للمستهلكين "
        "باستخدام بنية تحتية مستأجرة من مشغل شبكة أصلي بموجب اتفاقية جملة. "
        "من الناحية التنظيمية، يخضع المشغل الافتراضي لمتطلبات ترخيص مختلفة وأقل تعقيداً "
        "مقارنةً بالمشغل الأصلي، ولا يحتاج إلى ترخيص طيف ترددي. "
        "تختلف متطلبات ترخيص المشغلين الافتراضيين من دولة إلى أخرى في منطقة الخليج."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="TC-BENCH-003", lang="ar-en", domain="telecommunications",
    task_type="formal_correspondence", difficulty="basic",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write a formal Arabic email to a telecom vendor requesting a product demonstration. "
        "Core facts: Company wants to evaluate a specific telecom product. "
        "Required reasoning: Include product name, purpose, and timeline. "
        "Expected knowledge: Formal Arabic email structure; vendor communication. "
        "Risk: Using informal language in a formal vendor communication. "
        "Distinct from AC-TRAIN-001: Telecom vendor context, not general business meeting."
    ),
    prompt="Write a short formal email in Arabic to a telecom vendor requesting a product demonstration.",
    response=(
        "الموضوع: طلب عرض توضيحي لمنتج [اسم المنتج]\n\n"
        "السادة / فريق المبيعات\nتحية طيبة وبعد،\n\n"
        "يسرّنا التواصل معكم للاستفسار عن إمكانية ترتيب عرض توضيحي لمنتجكم [اسم المنتج]، "
        "لتقييم مدى ملاءمته لاحتياجات شركتنا التشغيلية في قطاع الاتصالات. "
        "نرجو إفادتنا بالمواعيد المتاحة خلال الأسبوعين القادمين.\n\n"
        "مع خالص التقدير،\n[الاسم] — [القسم]"
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="TC-BENCH-004", lang="en", domain="telecommunications",
    task_type="executive_briefing", difficulty="intermediate",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Brief on the commercial impact of Open RAN on GCC telecom operators. "
        "Core facts: Open RAN disaggregates radio access network hardware and software. "
        "Required reasoning: Explain the cost and vendor diversity benefits vs integration challenges. "
        "Expected knowledge: Disaggregation, multi-vendor, capex reduction, integration complexity. "
        "Risk: Describing Open RAN as only a cost-saving technology without the integration challenges. "
        "Distinct from TC-TRAIN-004: Open RAN technology, not 5G impact."
    ),
    prompt="Provide a brief executive briefing on the commercial impact of Open RAN technology on GCC telecom operators.",
    response=(
        "Open RAN (Open Radio Access Network) disaggregates the hardware and software components "
        "of radio access networks, enabling operators to mix equipment from multiple vendors. "
        "For GCC telecom operators, the potential commercial benefits include: "
        "(1) Reduced capital expenditure through competitive procurement across multiple vendors; "
        "(2) Reduced vendor dependency and improved negotiating leverage; and "
        "(3) Faster innovation cycles through software-defined network functions. "
        "However, significant challenges remain: "
        "(1) Integration complexity increases when combining equipment from different vendors; "
        "(2) Performance parity with traditional single-vendor RAN is not yet fully demonstrated; "
        "(3) Operational expertise requirements are higher. "
        "The commercial case for Open RAN in the GCC depends on each operator's specific "
        "vendor relationships and network modernisation timeline. "
        "Note: This briefing uses a fictional scenario for training purposes."
    ),
))

# ─── Government Regulation — Benchmark ────────────────────────────────────

BENCH_EXAMPLES.append(make_example(
    family="GR-BENCH-001", lang="en", domain="government_regulation",
    task_type="regulatory_explanation", difficulty="advanced",
    country="GCC", prov_type="SOURCE_DEPENDENT_FACTUAL",
    source_name="IMF Article IV Consultation Reports",
    source_url="https://www.imf.org/en/Publications/CR",
    scenario_brief=(
        "Task: Explain key IMF fiscal recommendations for GCC economies. "
        "Core facts: IMF Article IV consultations provide fiscal recommendations. "
        "Required reasoning: Identify three recurring themes in IMF GCC recommendations. "
        "Expected knowledge: Fiscal consolidation, non-oil revenue, subsidy reform. "
        "Risk: Citing specific recommendations from a specific year without noting they change. "
        "Distinct from GR-TRAIN-005: IMF recommendations, not Vision 2030."
    ),
    prompt="What key fiscal recommendations has the IMF typically made for GCC economies in Article IV consultations?",
    response=(
        "IMF Article IV consultations with GCC economies typically address three recurring themes: "
        "(1) Fiscal consolidation: Reducing fiscal deficits during periods of low oil prices "
        "through expenditure rationalisation and improved public financial management. "
        "(2) Non-oil revenue diversification: Expanding value-added tax, corporate income tax, "
        "and other non-hydrocarbon revenue sources to reduce dependence on oil revenues. "
        "(3) Subsidy reform: Rationalising energy and food subsidies to improve fiscal "
        "efficiency while protecting vulnerable populations through targeted transfers. "
        "Specific recommendations vary by country and year. For the most current recommendations, "
        "please consult the published Article IV consultation reports for each GCC country "
        "on the IMF website. This answer summarises general publicly available IMF guidance."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="GR-BENCH-002", lang="ar", domain="government_regulation",
    task_type="question_answering", difficulty="intermediate",
    country="Kuwait", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="Kuwait Ministry of Commerce", source_url="https://www.moci.gov.kw/",
    scenario_brief=(
        "Task: Summarize requirements for registering a foreign company in Kuwait. "
        "Core facts: Ministry of Commerce and Industry handles company registration. "
        "Required reasoning: Identify the four key requirements. "
        "Expected knowledge: Trade licence, commercial agent, regulatory approval, Kuwaitisation. "
        "Risk: Omitting the commercial agent requirement which is specific to Kuwait. "
        "Distinct from GR-TRAIN-001: Company registration, not procurement regulation."
    ),
    prompt="ما متطلبات تسجيل الشركات الأجنبية في الكويت وفق وزارة التجارة والصناعة؟",
    response=(
        "تستلزم تسجيل الشركات الأجنبية في الكويت استيفاء عدة متطلبات رئيسية: "
        "أولاً: تقديم طلب الترخيص التجاري مع المستندات التأسيسية المعتمدة والمصادق عليها. "
        "ثانياً: تعيين وكيل تجاري كويتي في بعض القطاعات وفق متطلبات قانون الوكالات التجارية. "
        "ثالثاً: الحصول على موافقة الجهات الرقابية المختصة حسب طبيعة النشاط التجاري. "
        "رابعاً: الالتزام بنسب الكويتة في التوظيف المحددة بموجب اللوائح العمالية. "
        "تتفاوت المتطلبات بحسب هيكل الملكية والقطاع والنشاط. "
        "يُنصح بمراجعة الموقع الرسمي لوزارة التجارة والصناعة للاطلاع على الإجراءات المحدّثة."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="GR-BENCH-003", lang="ar-en", domain="government_regulation",
    task_type="arabic_english_translation", difficulty="intermediate",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'Public-Private Partnership' into Arabic with a bilingual explanation. "
        "Core facts: PPPs are used in GCC infrastructure development. "
        "Required reasoning: Explain the risk-sharing mechanism between government and private sector. "
        "Expected knowledge: Risk allocation, concession period, government guarantee. "
        "Risk: Describing PPP as simply a government contract. "
        "Distinct from GR-TRAIN-006: PPP concept, not regulatory sandbox."
    ),
    prompt="Translate 'Public-Private Partnership (PPP)' into Arabic and explain how it works in GCC infrastructure development.",
    response=(
        "الترجمة: الشراكة بين القطاعين العام والخاص.\n\n"
        "الشرح بالعربية: الشراكة بين القطاعين العام والخاص هي ترتيب تعاقدي طويل الأمد تتعاون فيه "
        "الحكومة مع شركة خاصة لتطوير وتشغيل مشروع بنية تحتية، مع توزيع المخاطر والعوائد بينهما.\n\n"
        "English explanation: A Public-Private Partnership (PPP) is a long-term contractual "
        "arrangement in which a government entity collaborates with a private company to develop, "
        "finance, and operate an infrastructure project. The private partner typically finances "
        "and builds the asset, recovers its investment through user fees or government payments "
        "over a concession period, and transfers the asset to the government at the end. "
        "GCC governments have used PPPs for airports, hospitals, and utilities."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="GR-BENCH-004", lang="en", domain="government_regulation",
    task_type="hallucination_resistance", difficulty="basic",
    country="GCC", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to provide the exact current VAT rate in a specific GCC country without specifying which. "
        "Core facts: VAT rates vary by GCC country and have changed over time. "
        "Required reasoning: Explain that rates vary and direct to official sources. "
        "Expected knowledge: Saudi Arabia 15%, UAE 5%, Kuwait no VAT (as of knowledge cutoff). "
        "Risk: Providing a single rate that may not apply to the user's jurisdiction. "
        "Distinct from GR-TRAIN-007: VAT rate refusal, not data protection penalty."
    ),
    prompt="What is the current VAT rate in the GCC?",
    response=(
        "VAT rates vary significantly across GCC countries and have changed over time. "
        "There is no single GCC-wide VAT rate. "
        "As of my knowledge cutoff, Saudi Arabia applied a 15% VAT rate, the UAE applied 5%, "
        "and Kuwait had not yet implemented a general VAT. Other GCC countries have their own "
        "positions. Rates are subject to legislative change. "
        "For the current applicable rate in your specific jurisdiction, please consult the "
        "relevant national tax authority or a licensed tax advisor."
    ),
))

# ─── Energy and Logistics — Benchmark ─────────────────────────────────────

BENCH_EXAMPLES.append(make_example(
    family="EL-BENCH-001", lang="en", domain="energy_logistics",
    task_type="question_answering", difficulty="intermediate",
    country="GCC", prov_type="PUBLIC_RULE_SUMMARY",
    source_name="IEA", source_url="https://www.iea.org/",
    scenario_brief=(
        "Task: Explain the role of GCC sovereign wealth funds in energy transition investment. "
        "Core facts: GCC SWFs are increasingly investing in renewable energy. "
        "Required reasoning: Explain the strategic rationale for SWF investment in renewables. "
        "Expected knowledge: Diversification, long-term returns, energy transition risk. "
        "Risk: Describing SWF investment as purely financial without the strategic context. "
        "Distinct from EL-TRAIN-001: SWF energy transition investment, not LNG exports."
    ),
    prompt="What role are GCC sovereign wealth funds playing in energy transition investment?",
    response=(
        "GCC sovereign wealth funds are increasingly allocating capital to renewable energy "
        "and energy transition assets, driven by three strategic rationales: "
        "(1) Diversification: Reducing dependence on hydrocarbon revenues by building a "
        "portfolio of assets that perform well in a low-carbon future. "
        "(2) Long-term return potential: Renewable energy assets offer stable, long-duration "
        "cash flows that match the long-term investment horizon of sovereign funds. "
        "(3) Managing energy transition risk: GCC economies face structural risk from the "
        "global shift away from fossil fuels; investing in the transition itself hedges "
        "this risk. "
        "Specific investment allocations vary by fund and are subject to change. "
        "For current data, please consult the relevant fund's official publications."
    ),
    factual_dep=True,
))

BENCH_EXAMPLES.append(make_example(
    family="EL-BENCH-002", lang="ar", domain="energy_logistics",
    task_type="executive_briefing", difficulty="intermediate",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Brief on the strategic importance of the Strait of Hormuz for GCC energy exports. "
        "Core facts: Most GCC oil and LNG exports pass through the Strait of Hormuz. "
        "Required reasoning: Explain the chokepoint risk and mitigation strategies. "
        "Expected knowledge: Bypass pipelines, alternative routes, insurance implications. "
        "Risk: Describing the Strait as the only risk without mentioning mitigation. "
        "Distinct from EL-TRAIN-001: Chokepoint risk, not LNG capacity."
    ),
    prompt="أعدّ موجزاً تنفيذياً حول الأهمية الاستراتيجية لمضيق هرمز بالنسبة لصادرات الطاقة الخليجية.",
    response=(
        "يُمثّل مضيق هرمز الممر المائي الأكثر أهمية لصادرات النفط والغاز الخليجية، "
        "إذ يمر عبره جزء كبير من صادرات النفط الخام والغاز المسال لدول الخليج. "
        "يُشكّل هذا الاعتماد مخاطر جيوسياسية تتعلق بأي اضطراب في حركة الملاحة. "
        "تشمل استراتيجيات التخفيف: "
        "أولاً: خطوط الأنابيب البديلة كخط أنابيب أبوظبي الذي يتجاوز المضيق. "
        "ثانياً: الاحتياطيات الاستراتيجية لدى الدول المستوردة. "
        "ثالثاً: تنويع مسارات الشحن عند توافر بدائل. "
        "ملاحظة: هذا تحليل افتراضي لأغراض التدريب."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="EL-BENCH-003", lang="ar-en", domain="energy_logistics",
    task_type="arabic_english_translation", difficulty="basic",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate 'Upstream Oil Operations' into Arabic with a bilingual definition. "
        "Core facts: Upstream refers to exploration and production. "
        "Required reasoning: Distinguish upstream from midstream and downstream. "
        "Expected knowledge: Exploration, drilling, production; value chain stages. "
        "Risk: Describing upstream as including refining (which is downstream). "
        "Distinct from IF-TRAIN-009: Energy sector term, not Islamic finance."
    ),
    prompt="Translate the energy term 'Upstream Oil Operations' into Arabic and explain how it differs from midstream and downstream.",
    response=(
        "الترجمة: عمليات النفط الأولية (المنبع النفطي).\n\n"
        "التعريف بالعربية: تشمل عمليات المنبع النفطي مراحل الاستكشاف والحفر والإنتاج "
        "واستخراج النفط والغاز الخام من باطن الأرض أو قاع البحر.\n\n"
        "Distinction from other stages:\n"
        "Upstream: Exploration, drilling, and production of crude oil and natural gas.\n"
        "Midstream: Transportation, storage, and processing of crude oil and natural gas.\n"
        "Downstream: Refining crude oil into petroleum products and distributing to end users."
    ),
))

# ─── Executive Decision — Benchmark ───────────────────────────────────────

BENCH_EXAMPLES.append(make_example(
    family="ED-BENCH-001", lang="en", domain="executive_decision",
    task_type="decision_comparison", difficulty="advanced",
    country="GCC", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Compare greenfield investment versus joint venture for a fictional GCC market entry. "
        "Core facts: Fictional foreign company considering two entry modes. "
        "Required reasoning: Identify the trade-offs in control, speed, and local knowledge. "
        "Expected knowledge: Greenfield control vs JV local knowledge; regulatory requirements. "
        "Risk: Recommending one mode without acknowledging the sector-specific constraints. "
        "Distinct from ED-TRAIN-001: Market entry mode comparison, not organic vs acquisition."
    ),
    prompt="Compare greenfield investment versus a joint venture as market entry strategies for a fictional foreign company entering the GCC.",
    response=(
        "Greenfield investment provides full ownership and control, allowing the company to "
        "implement its own systems, culture, and processes without compromise. However, it "
        "requires more time and capital to establish, and lacks the local market knowledge "
        "and regulatory relationships that a local partner provides. "
        "A joint venture with a local partner accelerates market entry, provides access to "
        "established customer relationships and regulatory approvals, and may be required "
        "in sectors with foreign ownership restrictions. However, it involves shared control, "
        "potential misalignment of objectives, and governance complexity. "
        "Key considerations for the GCC context: "
        "(1) Some GCC sectors require local ownership participation, making JV the only option. "
        "(2) Government relationships are critical in the GCC; a well-connected local partner "
        "can significantly accelerate approvals and contract awards. "
        "(3) Cultural alignment between the partners is essential for JV success. "
        "Note: This analysis uses a fictional company for training purposes."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="ED-BENCH-002", lang="ar", domain="executive_decision",
    task_type="question_answering", difficulty="intermediate",
    country="Kuwait", prov_type="ORIGINAL_SCENARIO",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Identify key ESG considerations for a fictional Kuwaiti listed company. "
        "Core facts: Fictional company subject to Kuwait Stock Exchange ESG requirements. "
        "Required reasoning: Identify three ESG dimensions relevant to GCC listed companies. "
        "Expected knowledge: Environmental disclosure, social responsibility, governance standards. "
        "Risk: Applying European ESG standards directly without GCC context. "
        "Distinct from ED-TRAIN-003: ESG considerations, not family business risks."
    ),
    prompt="ما الاعتبارات الرئيسية لمعايير الحوكمة والمسؤولية الاجتماعية والبيئية (ESG) لشركة كويتية مدرجة افتراضية؟",
    response=(
        "تشمل الاعتبارات الرئيسية لمعايير ESG لشركة كويتية مدرجة افتراضية ثلاثة أبعاد: "
        "أولاً: البُعد البيئي — الإفصاح عن بصمة الكربون واستهلاك الطاقة والمياه، "
        "والالتزام بالمعايير البيئية المحلية والدولية. "
        "ثانياً: البُعد الاجتماعي — تطبيق سياسات عادلة في التوظيف والتدريب والسلامة المهنية، "
        "والمساهمة في برامج المسؤولية الاجتماعية المجتمعية. "
        "ثالثاً: بُعد الحوكمة — الالتزام بمعايير الحوكمة الصادرة عن هيئة أسواق المال الكويتية، "
        "بما تشمل استقلالية مجلس الإدارة وسياسات مكافحة الفساد والإفصاح المالي. "
        "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="ED-BENCH-003", lang="ar-en", domain="executive_decision",
    task_type="formal_correspondence", difficulty="intermediate",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Draft a bilingual Sukuk issuance announcement executive summary. "
        "Core facts: Company announcing a Sukuk issuance to investors. "
        "Required reasoning: Provide parallel Arabic and English content with key terms. "
        "Expected knowledge: Sukuk terminology in both languages; investor communication. "
        "Risk: Using different financial terms in the two language versions. "
        "Distinct from GB-BENCH-003: Sukuk announcement, not bank results."
    ),
    prompt="Draft a short bilingual (Arabic and English) executive summary for a Sukuk issuance announcement.",
    response=(
        "AR: ملخص تنفيذي — إصدار صكوك\n"
        "تُعلن [الشركة] عن إصدار صكوك إجارة بقيمة إجمالية تبلغ [المبلغ]، "
        "بعائد سنوي قدره [النسبة]%، وأجل استحقاق [المدة] سنوات. "
        "حصل الإصدار على موافقة الهيئة الشرعية وتصنيف ائتماني [التصنيف]. "
        "تُستخدم العائدات لتمويل [المشروع].\n\n"
        "EN: Executive Summary — Sukuk Issuance\n"
        "[Company] announces the issuance of Ijarah Sukuk totalling [Amount], "
        "with an annual profit rate of [Rate]% and a maturity of [Tenor] years. "
        "The issuance has received Sharia board approval and a credit rating of [Rating]. "
        "Proceeds will finance [Project].\n\n"
        "Note: Bracketed fields are placeholders; this is a training template only."
    ),
))

# ─── Arabic-English Correspondence — Benchmark ────────────────────────────

BENCH_EXAMPLES.append(make_example(
    family="AC-BENCH-001", lang="ar", domain="arabic_english_correspondence",
    task_type="formal_correspondence", difficulty="intermediate",
    country="Kuwait", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write a formal Arabic letter requesting a meeting with a government official. "
        "Core facts: Company seeking a meeting with a ministry official. "
        "Required reasoning: Use appropriate honorifics and formal structure. "
        "Expected knowledge: Formal Arabic letter structure; government correspondence conventions. "
        "Risk: Using business letter conventions inappropriate for government correspondence. "
        "Distinct from AC-TRAIN-001: Government official, not business meeting."
    ),
    prompt="اكتب رسالة رسمية باللغة العربية تطلب فيها موعداً مع مسؤول حكومي لمناقشة مشروع استثماري.",
    response=(
        "الموضوع: طلب موعد لمناقشة مشروع استثماري\n\n"
        "معالي / سعادة [اللقب والاسم]\nتحية طيبة وبعد،\n\n"
        "تتشرف شركتنا بالتواصل مع معاليكم لطلب موعد مناسب لعرض مشروعنا الاستثماري "
        "الذي يتوافق مع الأولويات التنموية لوزارتكم الموقرة. "
        "نأمل أن يتسع وقتكم الكريم لاستقبالنا في الوقت الذي تُحددونه.\n\n"
        "مع خالص الاحترام والتقدير،\n[الاسم] — [المسمى الوظيفي] — [اسم الشركة]"
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="AC-BENCH-002", lang="en", domain="arabic_english_correspondence",
    task_type="formal_correspondence", difficulty="intermediate",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Write a formal English letter of intent for a business acquisition. "
        "Core facts: Company expressing non-binding intent to acquire a target. "
        "Required reasoning: Include the key LOI elements: intent, price range, exclusivity, timeline. "
        "Expected knowledge: LOI structure; non-binding nature; key terms. "
        "Risk: Making the LOI sound binding when it should be non-binding. "
        "Distinct from AC-TRAIN-002: Letter of intent, not proposal declination."
    ),
    prompt="Write a formal English letter of intent for a non-binding business acquisition.",
    response=(
        "Subject: Letter of Intent — Proposed Acquisition of [Target Company]\n\n"
        "Dear [Name],\n\n"
        "This letter sets out the non-binding terms under which [Acquirer] proposes to "
        "acquire [Target Company]. "
        "We propose a purchase price in the range of [Amount], subject to due diligence "
        "and final agreement on terms. "
        "We request a period of exclusive negotiation of [Duration] to complete our review. "
        "This letter does not constitute a binding commitment and is subject to the execution "
        "of a definitive sale and purchase agreement.\n\n"
        "We look forward to your response.\n\n"
        "Yours sincerely,\n[Name] — [Title]"
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="AC-BENCH-003", lang="ar-en", domain="arabic_english_correspondence",
    task_type="arabic_english_translation", difficulty="advanced",
    country="GCC", prov_type="TRANSLATION_OR_CORRESPONDENCE",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Translate a formal Arabic board resolution excerpt into English. "
        "Core facts: Board resolution authorising a specific transaction. "
        "Required reasoning: Preserve the legal precision of the Arabic text. "
        "Expected knowledge: Board resolution language; legal translation conventions. "
        "Risk: Translating too loosely and losing the legal precision. "
        "Distinct from AC-TRAIN-008: Translating a board resolution, not a marketing phrase."
    ),
    prompt=(
        "Translate this formal Arabic board resolution excerpt into precise English:\n"
        "«قرر مجلس الإدارة بالإجماع تفويض الرئيس التنفيذي بالتوقيع على اتفاقية الشراكة "
        "مع الطرف الثاني، وذلك وفق الشروط والأحكام المعتمدة في هذا الاجتماع.»"
    ),
    response=(
        "Translation: 'The Board of Directors resolved unanimously to authorise the Chief "
        "Executive Officer to execute the partnership agreement with the second party, "
        "in accordance with the terms and conditions approved at this meeting.'\n\n"
        "Translation note: The Arabic 'بالإجماع' (unanimously) and 'وفق الشروط والأحكام المعتمدة' "
        "(in accordance with the approved terms and conditions) are legally significant phrases "
        "that have been preserved precisely in the English translation. "
        "In formal legal documents, 'تفويض' is rendered as 'authorise' rather than 'delegate' "
        "to reflect the full scope of the authority granted."
    ),
))

BENCH_EXAMPLES.append(make_example(
    family="AC-BENCH-004", lang="ar", domain="arabic_english_correspondence",
    task_type="refusal_missing_info", difficulty="basic",
    country="GCC", prov_type="SAFETY_OR_REFUSAL",
    source_name="AgenThink Internal", source_url="https://github.com/agenthinkai/jupiter-shot",
    scenario_brief=(
        "Task: Refuse to translate a document without seeing it. "
        "Core facts: Translation quality depends on the source text. "
        "Required reasoning: Explain what is needed before translation can begin. "
        "Expected knowledge: Source text, target language, formality level, purpose. "
        "Risk: Producing a generic template that does not match the actual document. "
        "Distinct from AC-TRAIN-007: Translation refusal, not letter-writing refusal."
    ),
    prompt="هل يمكنك ترجمة وثيقتي إلى الإنجليزية دون أن أُرسل لك النص؟",
    response=(
        "لا يمكنني إجراء أي ترجمة دون الاطلاع على النص الأصلي. "
        "لتقديم ترجمة دقيقة وملائمة، أحتاج إلى: "
        "أولاً: النص الكامل للوثيقة المراد ترجمتها. "
        "ثانياً: اللغة المستهدفة (الإنجليزية في هذه الحالة). "
        "ثالثاً: مستوى الرسمية المطلوب (رسمي، قانوني، تجاري، أو عام). "
        "رابعاً: الغرض من الترجمة حتى أتمكن من اختيار المصطلحات الأنسب للسياق. "
        "يُرجى إرسال النص وسأبدأ العمل فوراً."
    ),
))

# ═══════════════════════════════════════════════════════════════════════════
# DATASET BUILDER
# ═══════════════════════════════════════════════════════════════════════════

AUTHORIZED_TRAIN_COUNT = 600
AUTHORIZED_VALID_COUNT = 100
AUTHORIZED_EVAL_COUNT = 150

TRAIN_LANG_TARGETS = {"ar": 270, "en": 180, "ar-en": 150}
BENCH_LANG_TARGETS = {"ar": 113, "en": 75, "ar-en": 62}

TRAIN_DOMAIN_TARGETS = {
    "islamic_finance": 120, "gcc_banking": 90, "telecommunications": 90,
    "government_regulation": 90, "energy_logistics": 60,
    "executive_decision": 90, "arabic_english_correspondence": 60,
}
BENCH_DOMAIN_TARGETS = {
    "islamic_finance": 50, "gcc_banking": 38, "telecommunications": 38,
    "government_regulation": 38, "energy_logistics": 25,
    "executive_decision": 38, "arabic_english_correspondence": 23,
}


def build_record(
    example: Dict,
    split: str,
    idx: int,
    now: str,
) -> Dict[str, Any]:
    prompt = example["prompt"]
    response = example["response"]

    # Validate no artificial markers
    assert not has_artificial_marker(prompt), f"Artificial marker in prompt: {example['content_family_id']}"
    assert not has_artificial_marker(response), f"Artificial marker in response: {example['content_family_id']}"

    ar_count_p = ar_char_count(prompt)
    ar_count_r = ar_char_count(response)
    en_count_p = en_char_count(prompt)
    en_count_r = en_char_count(response)
    total_p = max(len(prompt.replace(" ", "")), 1)
    total_r = max(len(response.replace(" ", "")), 1)

    lang = example["language"]
    lang_contract_passed = check_language_contract(lang, prompt, response)

    prov_type = example["provenance_type"]
    prov_fields: Dict[str, Any] = {
        "provenance_type": prov_type,
        "factual_dependency": example.get("factual_dependency", False),
        "source_name": example["source_name"],
        "source_url": example["source_url"],
        "source_access_date": example.get("source_access_date", "2024-01-01"),
        "source_jurisdiction": example.get("country_or_region", "GCC"),
        "answer_relationship_to_source": "summarizes" if example.get("factual_dependency") else "original",
        "content_family_id": example["content_family_id"],
        "scenario_brief": example["scenario_brief"],
    }

    if prov_type == "ORIGINAL_SCENARIO":
        prov_fields["fictional_disclaimer"] = (
            "All companies, numbers, and situations in this example are fictional "
            "and created solely for training purposes."
        )

    return {
        "example_id": f"seed4b-{split}-{idx:04d}",
        "split": split,
        "prompt": prompt,
        "response": response,
        "language": lang,
        "prompt_language": "ar" if ar_count_p > en_count_p else ("ar-en" if ar_count_p > 5 and en_count_p > 5 else "en"),
        "response_language": "ar" if ar_count_r > en_count_r else ("ar-en" if ar_count_r > 5 and en_count_r > 5 else "en"),
        "language_pair": lang,
        "arabic_character_count_prompt": ar_count_p,
        "arabic_character_count_response": ar_count_r,
        "arabic_ratio_prompt": round(ar_count_p / total_p, 4),
        "arabic_ratio_response": round(ar_count_r / total_r, 4),
        "language_contract_passed": lang_contract_passed,
        "country_or_region": example.get("country_or_region", "GCC"),
        "domain": example["domain"],
        "task_type": example["task_type"],
        "difficulty": example["difficulty"],
        "source_type": "internally_authored",
        "licence_or_permission": "AgenThink Internal — Not for redistribution",
        "creation_method": "manual_authoring",
        "author_type": "internal_human_author",
        "factuality_review_status": "pending_human",
        "arabic_review_status": "pending_human",
        "human_review_required": True,
        "sensitive_data_status": "clean",
        "created_at": now,
        "modified_at": now,
        "inclusion_reason": "Internally authored GCC enterprise diagnostic example",
        "rejection_reason": None,
        "content_sha256": sha256_of(f"{prompt}\n{response}"),
        "canonical_prompt_hash": canonical_hash(prompt),
        "canonical_response_hash": canonical_hash(response),
        "canonical_pair_hash": canonical_hash(f"{prompt}\n{response}"),
        "scenario_brief_hash": sha256_of(canonicalize(example["scenario_brief"])),
        "schema_version": SCHEMA_VERSION,
        "dataset_version": DATASET_VERSION,
        **prov_fields,
    }


def write_split(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jupiter Seed 4B — Gold Dataset Builder v4")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("training/jupiter_seed_4b/data"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rng = random.Random(args.seed)

    log.info("Training examples in library: %d", len(TRAIN_EXAMPLES))
    log.info("Benchmark examples in library: %d", len(BENCH_EXAMPLES))

    # Validate no artificial markers in any example
    for ex in TRAIN_EXAMPLES + BENCH_EXAMPLES:
        assert not has_artificial_marker(ex["prompt"]), (
            f"Artificial marker in prompt: {ex['content_family_id']}"
        )
        assert not has_artificial_marker(ex["response"]), (
            f"Artificial marker in response: {ex['content_family_id']}"
        )

    # Validate no duplicate scenario briefs
    train_briefs = [canonicalize(ex["scenario_brief"]) for ex in TRAIN_EXAMPLES]
    bench_briefs = [canonicalize(ex["scenario_brief"]) for ex in BENCH_EXAMPLES]
    assert len(set(train_briefs)) == len(train_briefs), "Duplicate scenario briefs in training set"
    assert len(set(bench_briefs)) == len(bench_briefs), "Duplicate scenario briefs in benchmark set"

    # Validate no canonical response duplicates within each set
    train_resp_hashes = [canonical_hash(ex["response"]) for ex in TRAIN_EXAMPLES]
    bench_resp_hashes = [canonical_hash(ex["response"]) for ex in BENCH_EXAMPLES]
    assert len(set(train_resp_hashes)) == len(train_resp_hashes), "Duplicate canonical responses in training set"
    assert len(set(bench_resp_hashes)) == len(bench_resp_hashes), "Duplicate canonical responses in benchmark set"

    # Validate no cross-set canonical response duplicates
    cross_dups = set(train_resp_hashes) & set(bench_resp_hashes)
    assert not cross_dups, f"Cross-set canonical response duplicates: {len(cross_dups)} groups"

    # Validate content families do not cross train/bench boundary
    train_families = {ex["content_family_id"] for ex in TRAIN_EXAMPLES}
    bench_families = {ex["content_family_id"] for ex in BENCH_EXAMPLES}
    family_overlap = train_families & bench_families
    assert not family_overlap, f"Content families cross train/bench boundary: {family_overlap}"

    log.info("All pre-build validations passed.")

    # Build records
    train_records = [build_record(ex, "train", i, now) for i, ex in enumerate(TRAIN_EXAMPLES)]
    bench_records = [build_record(ex, "bench", i, now) for i, ex in enumerate(BENCH_EXAMPLES)]

    valid_records = bench_records[:len(bench_records)//2]
    eval_records = bench_records[len(bench_records)//2:]
    for i, r in enumerate(valid_records):
        r["example_id"] = f"seed4b-valid-{i:04d}"
        r["split"] = "valid"
    for i, r in enumerate(eval_records):
        r["example_id"] = f"seed4b-eval-{i:04d}"
        r["split"] = "eval"

    all_records = train_records + valid_records + eval_records
    total = len(all_records)

    # Honest count report
    log.info("=== HONEST DATASET SIZE REPORT ===")
    log.info("Authorized targets: train=%d, valid=%d, eval=%d, total=%d",
             AUTHORIZED_TRAIN_COUNT, AUTHORIZED_VALID_COUNT, AUTHORIZED_EVAL_COUNT,
             AUTHORIZED_TRAIN_COUNT + AUTHORIZED_VALID_COUNT + AUTHORIZED_EVAL_COUNT)
    log.info("Actual distinct examples: train=%d, valid=%d, eval=%d, total=%d",
             len(train_records), len(valid_records), len(eval_records), total)

    if total < AUTHORIZED_TRAIN_COUNT + AUTHORIZED_VALID_COUNT + AUTHORIZED_EVAL_COUNT:
        shortfall = (AUTHORIZED_TRAIN_COUNT + AUTHORIZED_VALID_COUNT + AUTHORIZED_EVAL_COUNT) - total
        log.warning("HONEST SIZE GATE: Dataset is %d examples short of the authorized target.", shortfall)
        log.warning("This is the honest count. Do not fabricate additional examples.")
        log.warning("Human authoring effort required to reach 850: approximately %d additional examples.", shortfall)

    # Language distribution
    def count_lang(records, lang):
        return sum(1 for r in records if r["language"] == lang)

    log.info("Train language: ar=%d en=%d ar-en=%d",
             count_lang(train_records, "ar"),
             count_lang(train_records, "en"),
             count_lang(train_records, "ar-en"))
    log.info("Bench language: ar=%d en=%d ar-en=%d",
             count_lang(valid_records + eval_records, "ar"),
             count_lang(valid_records + eval_records, "en"),
             count_lang(valid_records + eval_records, "ar-en"))

    # Write splits
    write_split(train_records, args.output_dir / "train.jsonl")
    write_split(valid_records, args.output_dir / "valid.jsonl")
    write_split(eval_records, args.output_dir / "eval.jsonl")

    # Human review queue: 50 examples covering all required categories
    queue: List[Dict] = []
    seen_domain_lang: Dict[str, int] = {}

    # Ensure SOURCE_DEPENDENT_FACTUAL examples are included
    sdf = [r for r in all_records if r.get("provenance_type") == "SOURCE_DEPENDENT_FACTUAL"]
    for r in sdf[:3]:
        queue.append(r)
        key = f"{r['domain']}-{r['language']}"
        seen_domain_lang[key] = seen_domain_lang.get(key, 0) + 1

    # Fill remaining slots
    shuffled = list(all_records)
    rng.shuffle(shuffled)
    for r in shuffled:
        if r in queue:
            continue
        key = f"{r['domain']}-{r['language']}"
        if seen_domain_lang.get(key, 0) < 3:
            queue.append(r)
            seen_domain_lang[key] = seen_domain_lang.get(key, 0) + 1
        if len(queue) >= min(50, total):
            break

    write_split(queue, args.output_dir / "human_review_queue.jsonl")
    log.info("Human review queue: %d examples", len(queue))

    # Final uniqueness check
    all_canonical_resp = [canonical_hash(r["response"]) for r in all_records]
    unique_canonical = len(set(all_canonical_resp))
    log.info("Unique canonical responses: %d / %d", unique_canonical, total)
    if unique_canonical < total:
        log.error("CANONICAL DUPLICATE RESPONSES DETECTED: %d duplicates", total - unique_canonical)
        sys.exit(1)
    log.info("All canonical responses are unique. PASS.")


if __name__ == "__main__":
    main()
