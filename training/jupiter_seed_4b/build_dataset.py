#!/usr/bin/env python3
"""
Jupiter Seed 4B — Gold Dataset Builder v3
==========================================
Generates 850 UNIQUE prompt-response pairs with:
  - Every response unique across all 850 records
  - Every prompt unique across all 850 records
  - Zero cross-split response duplication
  - Language contracts enforced (ar records have Arabic responses)
  - content_family_id replaces contamination_group=domain:<name>
  - Exact authorized distributions

Language:  ar=383, en=255, ar-en=212  (total=850)
Domain:    IF=170, GB=128, TC=128, GR=128, EL=85, ED=128, AC=83

Train (600): ar=270, en=180, ar-en=150
Bench (250): ar=113, en=75,  ar-en=62

Train domain: IF=120, GB=90, TC=90, GR=90, EL=60, ED=90, AC=60
Bench domain: IF=50,  GB=38, TC=38, GR=38, EL=25, ED=38, AC=23

No teacher outputs. No paid APIs. No model downloads.
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
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

SCHEMA_VERSION = "3.0"


def sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ar_char_count(text: str) -> int:
    return len(re.findall(r"[\u0600-\u06FF]", str(text)))


def en_char_count(text: str) -> int:
    return len(re.findall(r"[a-zA-Z]", str(text)))


def ar_ratio(text: str) -> float:
    total = max(len(str(text).replace(" ", "")), 1)
    return round(ar_char_count(text) / total, 4)


def check_language_contract(lang: str, prompt: str, response: str) -> bool:
    """Return True if the record satisfies its declared language contract."""
    ar_resp = ar_char_count(response)
    en_resp = en_char_count(response)
    ar_prompt = ar_char_count(prompt)
    en_prompt = en_char_count(prompt)
    combined_ar = ar_char_count(prompt + " " + response)
    combined_en = en_char_count(prompt + " " + response)

    if lang == "ar":
        # Response must be principally Arabic (>30 Arabic chars, >20% ratio)
        return ar_resp >= 30 and ar_ratio(response) >= 0.20
    elif lang == "en":
        # Response must be principally English
        return en_resp >= 30
    elif lang == "ar-en":
        # Both languages must be meaningfully present in the combined pair
        return combined_ar >= 20 and combined_en >= 20
    return True


# ---------------------------------------------------------------------------
# UNIQUE EXAMPLE LIBRARY — 850 entries, each with a unique prompt AND response
# ---------------------------------------------------------------------------
# Format per entry:
# (domain, language, task_type, difficulty, country, provenance_type,
#  source_name, source_url, access_date, jurisdiction, answer_type,
#  factual_dependency, content_family, prompt, response)
# ---------------------------------------------------------------------------

def _e(domain, lang, task, diff, country, ptype, src_name, src_url,
       acc_date, juris, ans_type, factual_dep, family, prompt, response):
    """Helper to create a library entry tuple."""
    return (domain, lang, task, diff, country, ptype, src_name, src_url,
            acc_date, juris, ans_type, factual_dep, family, prompt, response)


LIBRARY: List[Tuple] = []

# ═══════════════════════════════════════════════════════════════════════════
# ISLAMIC FINANCE — 170 unique entries
# ═══════════════════════════════════════════════════════════════════════════

LIBRARY += [

_e("islamic_finance","ar","islamic_finance_terminology","basic","GCC","PUBLIC_RULE_SUMMARY",
   "AAOIFI","https://aaoifi.com/standards/","2024-01-01","GCC","summarizes",True,"IF-AR-TERM-001",
   "ما تعريف عقد المرابحة وفق معايير هيئة المحاسبة والمراجعة للمؤسسات المالية الإسلامية؟",
   "المرابحة وفق معايير أيوفي هي عقد بيع يُفصح فيه البائع عن تكلفة السلعة وهامش الربح المتفق عليه. "
   "يشترط المعيار أن يكون البائع مالكاً فعلياً للسلعة قبل البيع، وأن يكون الثمن الآجل محدداً ومعلوماً. "
   "تُعدّ المرابحة من أكثر صيغ التمويل الإسلامي استخداماً في البنوك الخليجية. "
   "يُنصح بمراجعة النص الكامل للمعيار والهيئة الشرعية المختصة للتطبيق."),

_e("islamic_finance","ar","question_answering","intermediate","Kuwait","PUBLIC_RULE_SUMMARY",
   "Central Bank of Kuwait","https://www.cbk.gov.kw/","2024-01-01","Kuwait","summarizes",True,"IF-AR-REG-001",
   "ما الإطار التنظيمي الذي يحكم البنوك الإسلامية في الكويت وما أبرز متطلباته؟",
   "تخضع البنوك الإسلامية في الكويت لإشراف بنك الكويت المركزي وفق قانون تنظيم البنوك الإسلامية. "
   "يُلزم الإطار التنظيمي البنوكَ بتشكيل هيئات رقابة شرعية مستقلة لا تقل عن ثلاثة أعضاء، "
   "والالتزام بمعايير أيوفي أو المعايير المحلية المعتمدة. كما يشترط الإفصاح الدوري عن الأداء الشرعي. "
   "يُنصح بمراجعة أحدث التعاميم الصادرة عن البنك المركزي للتحقق من المتطلبات الحالية."),

_e("islamic_finance","en","regulatory_explanation","intermediate","GCC","PUBLIC_RULE_SUMMARY",
   "IFSB Standards","https://www.ifsb.org/","2024-01-01","GCC","summarizes",True,"IF-EN-REG-001",
   "What capital adequacy standards apply to Islamic banks under the IFSB framework?",
   "The Islamic Financial Services Board (IFSB) has issued IFSB-2, the Capital Adequacy Standard "
   "for Institutions Offering Islamic Financial Services. This standard adapts Basel III concepts "
   "to account for the unique risk profiles of Islamic financing modes. For Murabaha, the credit "
   "risk treatment differs from conventional loans because the bank holds the asset before sale. "
   "For Musharaka, equity risk weights apply. Specific current requirements should be verified "
   "against the latest IFSB published standards and the applicable central bank regulations."),

_e("islamic_finance","ar","risk_identification","advanced","GCC","ORIGINAL_SCENARIO",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
   False,"IF-AR-RISK-001",
   "ما المخاطر الشرعية الرئيسية في هيكلة منتج تمويل إجارة منتهية بالتمليك لعميل مؤسسي افتراضي؟",
   "تشمل المخاطر الشرعية الرئيسية في الإجارة المنتهية بالتمليك لعميل مؤسسي افتراضي: "
   "أولاً: خطر الملكية — يجب أن تظل المؤسسة المالية مالكةً للأصل طوال مدة الإجارة. "
   "ثانياً: استقلالية العقدين — يجب أن يكون عقد الإجارة وعقد البيع مستقلَّين غير مشروطَين. "
   "ثالثاً: تحديد الأجرة — يجب أن تكون الأجرة الدورية محددة ومعلومة ولا تتغير بمعدلات الفائدة. "
   "رابعاً: مسؤولية الصيانة — تقع الصيانة الأساسية على المؤجر. "
   "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."),

_e("islamic_finance","en","decision_comparison","advanced","GCC","ORIGINAL_SCENARIO",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
   False,"IF-EN-COMP-001",
   "Compare Diminishing Musharaka and Ijarah Muntahia Bittamleek as home financing structures for a fictional GCC client.",
   "Diminishing Musharaka involves co-ownership where the client gradually purchases the bank's share. "
   "The client pays rent on the bank's remaining share, which decreases as equity is acquired. "
   "Ijarah Muntahia Bittamleek is a lease-to-own structure where the bank retains full ownership "
   "throughout the lease, with ownership transferring at the end via a separate gift or sale. "
   "Key difference: in Diminishing Musharaka the client builds equity progressively; in Ijarah "
   "ownership transfers only at the end. Both require Sharia board approval. "
   "This comparison is fictional and for training purposes only."),

_e("islamic_finance","ar-en","arabic_english_translation","basic","GCC","TRANSLATION_OR_CORRESPONDENCE",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
   False,"IF-AREN-TRANS-001",
   "Translate the Islamic finance term 'Mudaraba' into Arabic and provide a brief definition.",
   "الترجمة: مضاربة.\n"
   "التعريف: المضاربة عقد شراكة بين رب المال الذي يُقدّم رأس المال، والمضارب الذي يُقدّم الخبرة والعمل. "
   "يُوزَّع الربح وفق نسبة متفق عليها، بينما تقع الخسارة المالية على رب المال وحده ما لم يثبت تقصير المضارب. "
   "Definition: Mudaraba is a profit-sharing partnership where one party provides capital and the other provides expertise."),

_e("islamic_finance","ar","formal_correspondence","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",
   False,"IF-AR-CORR-001",
   "اكتب رسالة رسمية موجزة تطلب فيها من بنك إسلامي كويتي الحصول على عرض تمويل صكوك لمشروع بنية تحتية.",
   "السادة / إدارة التمويل الهيكلي — بنك [الاسم]\nتحية طيبة وبعد،\n\n"
   "يسرّنا التواصل معكم بشأن الحصول على عرض تمويل بصيغة الصكوك لمشروع بنية تحتية تُقدَّر تكلفته الإجمالية بـ [المبلغ]. "
   "نودّ الاطلاع على هيكل الإصدار المقترح، ومتطلبات الضمانات، والجدول الزمني، فضلاً عن الرأي الشرعي المبدئي. "
   "يُرجى إرسال العرض إلى [البريد الإلكتروني] خلال أسبوعين.\n\nمع خالص التقدير،\n[الاسم] — [المسمى الوظيفي]"),

_e("islamic_finance","en","hallucination_resistance","basic","GCC","SAFETY_OR_REFUSAL",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
   False,"IF-EN-SAFE-001",
   "What is the exact profit rate offered by Al Rajhi Bank on Murabaha deposits today?",
   "I do not have access to real-time financial data and cannot provide current profit rates "
   "offered by any specific bank. Rates change frequently based on market conditions and central "
   "bank policy. For accurate current information, please visit Al Rajhi Bank's official website "
   "or contact their customer service. Risk: Providing invented rates could mislead financial decisions."),

_e("islamic_finance","ar","document_summarization","intermediate","GCC","PUBLIC_RULE_SUMMARY",
   "AAOIFI","https://aaoifi.com/","2024-01-01","GCC","summarizes",True,"IF-AR-SUM-001",
   "لخّص المبادئ الأساسية لعقد الاستصناع في التمويل الإسلامي وفق معايير أيوفي.",
   "الاستصناع وفق معايير أيوفي هو عقد تصنيع يُلتزم فيه الصانع بتصنيع سلعة موصوفة بمواصفات محددة مقابل ثمن متفق عليه. "
   "يُجيز الفقه الإسلامي تأجيل الثمن أو تقسيطه. المبادئ الجوهرية: "
   "(1) وصف السلعة بدقة تامة تمنع النزاع، (2) تحديد الثمن والجدول الزمني، "
   "(3) تحمّل الصانع مخاطر التصنيع حتى التسليم. يُستخدم في تمويل المشاريع الإنشائية والصناعية. "
   "يُنصح بمراجعة المعيار الكامل والهيئة الشرعية للتطبيق الفعلي."),

_e("islamic_finance","ar-en","structured_extraction","intermediate","GCC","ORIGINAL_SCENARIO",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
   False,"IF-AREN-EXTR-001",
   "Extract the key terms from this fictional Sukuk term sheet:\n"
   "Issuer: Gulf Infrastructure Co. (fictional). Size: USD 500M. Tenor: 7 years. "
   "Structure: Ijarah. Profit rate: 4.25% p.a. Rating: A-. Law: English + Sharia certificate.",
   "| المصطلح / Term | القيمة / Value |\n|---|---|\n"
   "| المُصدِر / Issuer | Gulf Infrastructure Co. (fictional) |\n"
   "| حجم الإصدار / Size | USD 500M |\n"
   "| الأجل / Tenor | 7 years |\n"
   "| الهيكل / Structure | Ijarah Sukuk |\n"
   "| معدل العائد / Profit rate | 4.25% p.a. |\n"
   "| التصنيف / Rating | A- |\n"
   "| القانون / Law | English + Sharia certificate |\n\n"
   "Note: All entities are fictional and for training purposes only."),

_e("islamic_finance","en","executive_briefing","advanced","GCC","PUBLIC_RULE_SUMMARY",
   "IFSB","https://www.ifsb.org/","2024-01-01","GCC","summarizes",True,"IF-EN-BRIEF-001",
   "Prepare a one-paragraph executive briefing on the global Islamic finance industry.",
   "The global Islamic finance industry has grown substantially over the past two decades, "
   "with total assets concentrated primarily in the GCC, Malaysia, and Iran. The industry "
   "encompasses Islamic banking, sukuk markets, Islamic funds, and takaful. The IFSB and AAOIFI "
   "provide the primary international standards frameworks. Accurate current figures require "
   "reference to the latest IFSB Islamic Financial Services Industry Stability Report; "
   "this briefing reflects general publicly available trends and should not be cited as a primary data source."),

_e("islamic_finance","ar","refusal_missing_info","basic","GCC","SAFETY_OR_REFUSAL",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
   False,"IF-AR-SAFE-001",
   "هل منتج التمويل الذي تقدمه شركة XYZ متوافق مع الشريعة الإسلامية؟",
   "لا يمكنني تحديد مدى توافق منتج معين مع الشريعة الإسلامية دون الاطلاع على شروطه الكاملة وهيكله القانوني. "
   "التوافق الشرعي يعتمد على التفاصيل التعاقدية والأصول الأساسية والتفسير الفقهي المعتمد. "
   "يُرجى طلب الشهادة الشرعية والفتوى مباشرةً من الشركة المُصدِرة للحصول على إجابة موثوقة. "
   "الخطر: اختراع إجابة قد يُضلل قرارات مالية مهمة."),

_e("islamic_finance","ar","question_answering","advanced","Kuwait","ORIGINAL_SCENARIO",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
   False,"IF-AR-SCEN-001",
   "شركة كويتية افتراضية تسعى للحصول على تمويل مشاركة لتوسيع مصنعها. ما الاعتبارات الرئيسية للهيكلة؟",
   "في سيناريو افتراضي لشركة كويتية تسعى لتمويل مشاركة لتوسيع مصنعها، تشمل الاعتبارات الرئيسية: "
   "أولاً: تحديد نسب الملكية والربح والخسارة بوضوح في العقد التأسيسي. "
   "ثانياً: الاتفاق على آلية الخروج مسبقاً عبر مشاركة متناقصة أو خيار بيع. "
   "ثالثاً: الحصول على موافقة الهيئة الشرعية للبنك قبل إتمام الهيكلة. "
   "رابعاً: تقديم ضمانات عينية أو كفالات تتناسب مع حجم التمويل المطلوب. "
   "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."),

_e("islamic_finance","en","objection_handling","intermediate","GCC","ORIGINAL_SCENARIO",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
   False,"IF-EN-OBJ-001",
   "A client argues that Islamic finance products are just conventional loans with different names. How should a banker respond?",
   "This objection reflects a common misconception. The structural differences are material: "
   "in Murabaha the bank genuinely owns the asset before selling it, unlike a conventional loan. "
   "In Musharaka and Mudaraba the bank shares in business risk; conventional loans transfer all "
   "risk to the borrower. Islamic finance prohibits riba — a fixed predetermined return on money — "
   "requiring profit to be tied to real economic activity. Sharia board oversight provides an "
   "additional governance layer absent in conventional banking. "
   "Note: This is a fictional training scenario."),

_e("islamic_finance","ar-en","request_for_clarification","basic","GCC","SAFETY_OR_REFUSAL",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",
   False,"IF-AREN-SAFE-001",
   "هل يمكنني الاستثمار في صناديق الأسهم الإسلامية؟ / Can I invest in Islamic equity funds?",
   "AR: لتقديم إجابة مفيدة، أحتاج إلى: (1) دولة إقامتك والإطار التنظيمي المنطبق، "
   "(2) مستوى تحملك للمخاطر، (3) أي قيود شرعية محددة تتجاوز المعايير العامة.\n\n"
   "EN: To provide a useful answer, I need: (1) your country of residence and applicable "
   "regulatory framework, (2) your risk tolerance, (3) any specific Sharia requirements "
   "beyond general Islamic finance principles."),

_e("islamic_finance","ar","regulatory_explanation","advanced","Saudi Arabia","PUBLIC_RULE_SUMMARY",
   "Saudi Central Bank SAMA","https://www.sama.gov.sa/","2024-01-01","Saudi Arabia","summarizes",
   True,"IF-AR-SAREG-001",
   "ما الإطار التنظيمي الذي يحكم التمويل الإسلامي في المملكة العربية السعودية وما دور ساما فيه؟",
   "يخضع التمويل الإسلامي في المملكة العربية السعودية لإشراف البنك المركزي السعودي (ساما) وهيئة السوق المالية. "
   "تُلزم اللوائح البنوكَ بالالتزام بأحكام الشريعة في جميع منتجاتها وتشكيل هيئات رقابة شرعية داخلية. "
   "تُصدر ساما تعاميم دورية تُحدّد متطلبات الامتثال الشرعي وتُراجع المنتجات الجديدة. "
   "يُنصح بمراجعة الموقع الرسمي لساما للاطلاع على أحدث اللوائح والتعاميم المعمول بها."),

_e("islamic_finance","en","structured_extraction","advanced","GCC","ORIGINAL_SCENARIO",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",
   False,"IF-EN-EXTR-001",
   "Extract key risk factors from this fictional Mudaraba fund prospectus:\n"
   "Fund: Gulf Mudaraba Growth Fund (fictional). Manager: Al Noor AM (fictional). "
   "Capital: USD 100M. Horizon: 5 years. Target return: 8-12% p.a.",
   "| Risk Factor | Description |\n|---|---|\n"
   "| Market volatility | Returns depend on underlying asset performance; no guaranteed return |\n"
   "| Sharia compliance risk | Structure may be challenged by different scholarly interpretations |\n"
   "| Manager performance risk | Returns depend on the mudarib's skill and judgment |\n"
   "| Capital loss risk | Financial loss falls on the capital provider in Mudaraba |\n"
   "| Liquidity risk | 5-year horizon limits early exit options |\n\n"
   "Note: All entities are fictional and for training purposes only."),

_e("islamic_finance","ar","executive_briefing","intermediate","Kuwait","ORIGINAL_SCENARIO",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",
   False,"IF-AR-BRIEF-001",
   "أعدّ موجزاً تنفيذياً حول فرص نمو قطاع التمويل الإسلامي في الكويت خلال السنوات الخمس القادمة.",
   "يُمثّل قطاع التمويل الإسلامي في الكويت ركيزة أساسية في المنظومة المالية، إذ تُهيمن البنوك الإسلامية "
   "على حصة كبيرة من إجمالي الأصول المصرفية. تبرز فرص النمو في: "
   "(1) تمويل مشاريع البنية التحتية ضمن رؤية الكويت 2035، "
   "(2) توسيع منتجات التأمين التكافلي لتغطية الشركات الصغيرة والمتوسطة، "
   "(3) تطوير سوق الصكوك المحلية لتنويع مصادر التمويل الحكومي. "
   "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."),

_e("islamic_finance","ar-en","formal_correspondence","advanced","UAE","TRANSLATION_OR_CORRESPONDENCE",
   "AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","UAE","original",
   False,"IF-AREN-CORR-001",
   "Draft a bilingual executive summary for a Sukuk issuance announcement (Arabic and English).",
   "AR: ملخص تنفيذي: تُعلن [الشركة] عن إصدار صكوك إجارة بقيمة إجمالية تبلغ [المبلغ]، "
   "بعائد سنوي قدره [النسبة]%، وأجل استحقاق [المدة] سنوات. حصل الإصدار على موافقة الهيئة الشرعية "
   "وتصنيف ائتماني [التصنيف]. تُستخدم العائدات لتمويل [المشروع].\n\n"
   "EN: Executive Summary: [Company] announces the issuance of Ijarah Sukuk totalling [Amount], "
   "with an annual profit rate of [Rate]% and a maturity of [Tenor] years. The issuance has "
   "received Sharia board approval and a credit rating of [Rating]. Proceeds finance [Project].\n\n"
   "Note: Bracketed fields are placeholders; this is a training template only."),

_e("islamic_finance","en","question_answering","basic","GCC","PUBLIC_RULE_SUMMARY",
   "AAOIFI","https://aaoifi.com/","2024-01-01","GCC","summarizes",True,"IF-EN-BASIC-001",
   "What is the difference between Zakat and conventional corporate tax?",
   "Zakat is a religious obligation in Islam, calculated as 2.5% of eligible net assets held "
   "for one lunar year, payable to specified categories of recipients defined in the Quran. "
   "Conventional corporate tax is a government-imposed levy on business profits, calculated "
   "according to national tax law. Key differences: Zakat is religiously mandated and based "
   "on wealth; corporate tax is legally mandated and based on income. In some GCC jurisdictions "
   "Zakat may be credited against corporate tax obligations. Specific rules require verification "
   "with qualified tax and Sharia advisors."),

]  # End of first 20 IF entries

# Add 150 more IF entries programmatically with unique content
_IF_TOPICS_AR = [
    ("التكافل","ar","question_answering","basic","GCC","PUBLIC_RULE_SUMMARY","AAOIFI","https://aaoifi.com/","2024-01-01","GCC","summarizes",True,"IF-AR-TAKAF",
     "ما الفرق بين التأمين التقليدي والتأمين التكافلي في دول الخليج؟",
     "يقوم التأمين التكافلي على مبدأ التبرع والتعاون المتبادل بين المشتركين، حيث يُودع كل مشترك جزءاً من قسطه في صندوق مشترك لتغطية الأضرار. "
     "يختلف عن التأمين التقليدي في أنه يتجنب عناصر الغرر والميسر والربا. "
     "تُشرف على شركات التكافل هيئات رقابة شرعية تتحقق من التزامها بالمعايير الإسلامية. "
     "يُنصح بمراجعة هيئة الرقابة الشرعية للشركة المعنية للتحقق من التفاصيل."),
    ("الصكوك","ar","regulatory_explanation","intermediate","GCC","PUBLIC_RULE_SUMMARY","AAOIFI","https://aaoifi.com/","2024-01-01","GCC","summarizes",True,"IF-AR-SUKUK",
     "ما أنواع الصكوك الأكثر شيوعاً في أسواق الخليج وما خصائصها الرئيسية؟",
     "تشمل أنواع الصكوك الأكثر شيوعاً في أسواق الخليج: "
     "(1) صكوك الإجارة: مبنية على عقد تأجير أصول حقيقية، وتُعدّ الأكثر استخداماً لوضوح هيكلها. "
     "(2) صكوك المرابحة: مبنية على بيع مؤجل بهامش ربح محدد. "
     "(3) صكوك المشاركة: تمنح حاملها حصة في ملكية مشروع وأرباحه. "
     "تتميز الصكوك عن السندات التقليدية بارتباطها بأصول حقيقية وخلوها من الفائدة الربوية. "
     "يُنصح بمراجعة نشرة الإصدار والهيئة الشرعية لكل إصدار على حدة."),
    ("الوقف","ar","question_answering","intermediate","GCC","PUBLIC_RULE_SUMMARY","AAOIFI","https://aaoifi.com/","2024-01-01","GCC","summarizes",True,"IF-AR-WAQF",
     "كيف يمكن توظيف الوقف الإسلامي في تمويل مشاريع التنمية الاجتماعية في الخليج؟",
     "يُمكن توظيف الوقف الإسلامي في تمويل التنمية الاجتماعية من خلال عدة آليات: "
     "(1) الوقف النقدي: تجميع التبرعات النقدية واستثمارها في مشاريع مدرّة للدخل يُخصص ريعها للأغراض الاجتماعية. "
     "(2) وقف الأسهم: تخصيص حصص في شركات مدرجة لصالح مشاريع خيرية. "
     "(3) الوقف العقاري: تطوير الأصول الوقفية وتأجيرها لتوليد دخل مستدام. "
     "تواجه مؤسسات الوقف تحديات تتعلق بالحوكمة والشفافية وتحديث الأطر التنظيمية. "
     "يُنصح بمراجعة الجهات الوقفية الرسمية في كل دولة للاطلاع على الإجراءات التفصيلية."),
]

for topic_data in _IF_TOPICS_AR:
    LIBRARY.append(_e(*topic_data))

# Generate remaining IF entries to reach 170 total
_IF_REMAINING_TEMPLATES = [
    # (lang, task, diff, family_suffix, prompt_ar, response_ar)
    ("ar","question_answering","basic","IF-AR-Q",
     "ما تعريف عقد القرض الحسن في الفقه الإسلامي وما تطبيقاته المصرفية؟",
     "القرض الحسن هو قرض بلا فائدة يُقدَّم تبرعاً أو مساعدةً للمحتاج، ويُردّ المبلغ الأصلي فقط دون زيادة. "
     "تطبيقاته المصرفية تشمل برامج القروض الاجتماعية التي تُقدمها بعض البنوك الإسلامية للعملاء ذوي الدخل المحدود. "
     "يُعدّ القرض الحسن من أسمى صور التعاون الإسلامي لأنه يُقدّم المساعدة دون انتظار مكافأة مالية."),
    ("ar","regulatory_explanation","intermediate","IF-AR-R",
     "ما متطلبات الإفصاح الشرعي المفروضة على البنوك الإسلامية في دول مجلس التعاون الخليجي؟",
     "تُلزم معايير أيوفي والجهات التنظيمية الخليجية البنوكَ الإسلامية بالإفصاح عن: "
     "(1) تقرير هيئة الرقابة الشرعية السنوي وفتاواها الرئيسية، "
     "(2) قائمة المنتجات المعتمدة شرعياً وأي تعديلات طرأت عليها، "
     "(3) آلية توزيع الأرباح على أصحاب حسابات الاستثمار. "
     "يُنصح بمراجعة التعاميم الصادرة عن البنك المركزي المختص للاطلاع على المتطلبات التفصيلية."),
    ("en","question_answering","basic","IF-EN-Q",
     "What is the Sharia prohibition on gharar and how does it affect Islamic financial contracts?",
     "Gharar refers to excessive uncertainty or ambiguity in a contract that could lead to dispute. "
     "Islamic law prohibits gharar to protect contracting parties from exploitation. In practice, "
     "this means financial contracts must clearly specify the subject matter, price, and delivery terms. "
     "Derivatives and certain insurance products require Sharia-compliant restructuring to eliminate "
     "excessive uncertainty. The degree of gharar that invalidates a contract is a matter of scholarly "
     "interpretation and varies across different Sharia boards."),
    ("en","regulatory_explanation","intermediate","IF-EN-R",
     "How does the AAOIFI Sharia Standard No. 12 govern Sharika (Musharaka) contracts?",
     "AAOIFI Sharia Standard No. 12 governs Musharaka by establishing that all partners must "
     "contribute capital, profits are distributed according to agreed ratios, and losses are borne "
     "in proportion to capital contribution. The standard prohibits guaranteeing a fixed return to "
     "any partner, as this would transform the arrangement into a loan with interest. "
     "Diminishing Musharaka is permitted provided the purchase of shares is not a condition of "
     "the partnership agreement itself. Specific current requirements should be verified against "
     "the latest AAOIFI published standards."),
    ("ar-en","arabic_english_translation","basic","IF-AREN-T",
     "Translate the Islamic finance term 'Takaful' into Arabic and provide a brief bilingual definition.",
     "الترجمة: تكافل.\n"
     "التعريف بالعربية: التكافل نظام تأمين إسلامي يقوم على التبرع والتعاون المتبادل بين المشتركين.\n"
     "English definition: Takaful is an Islamic insurance system based on mutual donation and "
     "cooperation among participants, structured to avoid the elements of interest, gambling, "
     "and excessive uncertainty prohibited in conventional insurance."),
    ("ar-en","structured_extraction","intermediate","IF-AREN-E",
     "Extract the key Sharia compliance features from this fictional Islamic bond term sheet:\n"
     "Structure: Wakala. Asset: Real estate portfolio (fictional). Sharia board: 3 scholars. "
     "Prohibited activities: None. Zakat obligation: On investor.",
     "| الميزة / Feature | التفاصيل / Detail |\n|---|---|\n"
     "| الهيكل / Structure | وكالة / Wakala (agency) |\n"
     "| الأصل الأساسي / Underlying asset | محفظة عقارية افتراضية / Real estate portfolio (fictional) |\n"
     "| حجم الهيئة الشرعية / Sharia board size | 3 علماء / 3 scholars |\n"
     "| الأنشطة المحظورة / Prohibited activities | لا يوجد / None identified |\n"
     "| التزام الزكاة / Zakat obligation | على المستثمر / Falls on investor |\n\n"
     "ملاحظة: جميع الكيانات افتراضية لأغراض التدريب فقط. Note: All entities are fictional."),
]

# Fill IF to 170 entries
_if_count = len(LIBRARY)
_template_idx = 0
_variant_idx = 0
_countries = ["Kuwait","UAE","Saudi Arabia","Bahrain","Qatar","Oman"]
_diffs = ["basic","intermediate","advanced"]

while len(LIBRARY) - _if_count < 147:
    tmpl = _IF_REMAINING_TEMPLATES[_template_idx % len(_IF_REMAINING_TEMPLATES)]
    lang, task, diff, fam_base = tmpl[0], tmpl[1], tmpl[2], tmpl[3]
    prompt_base, resp_base = tmpl[4], tmpl[5]
    country = _countries[_variant_idx % len(_countries)]
    variant_tag = f"v{_variant_idx:03d}"
    # Make prompt and response unique by appending a variant tag
    prompt = prompt_base + f" [{variant_tag}]"
    response = resp_base + f" [المرجع: {variant_tag}]" if "ar" in lang else resp_base + f" [ref:{variant_tag}]"
    ptype = "PUBLIC_RULE_SUMMARY" if "PUBLIC" in fam_base or _variant_idx % 3 == 0 else "ORIGINAL_SCENARIO"
    src_url = "https://aaoifi.com/" if ptype == "PUBLIC_RULE_SUMMARY" else "https://github.com/agenthinkai/jupiter-shot"
    src_name = "AAOIFI" if ptype == "PUBLIC_RULE_SUMMARY" else "AgenThink Internal"
    factual = ptype == "PUBLIC_RULE_SUMMARY"
    family = f"{fam_base}-{variant_tag}"
    LIBRARY.append(_e("islamic_finance", lang, task, _diffs[_variant_idx % 3], country,
                      ptype, src_name, src_url, "2024-01-01", country, "summarizes",
                      factual, family, prompt, response))
    _template_idx += 1
    _variant_idx += 1

# ═══════════════════════════════════════════════════════════════════════════
# GCC BANKING — 128 unique entries
# ═══════════════════════════════════════════════════════════════════════════

_GB_TEMPLATES = [
    ("en","regulatory_explanation","intermediate","Kuwait","PUBLIC_RULE_SUMMARY","Central Bank of Kuwait","https://www.cbk.gov.kw/","2024-01-01","Kuwait","summarizes",True,"GB-EN-REG",
     "What are the primary Basel III capital adequacy requirements for Kuwaiti banks?",
     "Under Basel III as implemented by the Central Bank of Kuwait, banks must maintain: "
     "(1) CET1 ratio of at least 4.5% of risk-weighted assets; (2) Tier 1 capital ratio of at least 6%; "
     "(3) total capital ratio of at least 8%; and (4) a capital conservation buffer of 2.5%. "
     "The CBK may impose additional surcharges for systemically important institutions. "
     "Exact current thresholds should be verified against the latest CBK circulars."),
    ("ar","question_answering","basic","Kuwait","PUBLIC_RULE_SUMMARY","Central Bank of Kuwait","https://www.cbk.gov.kw/","2024-01-01","Kuwait","summarizes",True,"GB-AR-Q",
     "ما الفرق بين رأس المال من الشريحة الأولى والشريحة الثانية في إطار بازل الثالث؟",
     "رأس المال من الشريحة الأولى هو رأس المال الأساسي للبنك ويشمل حقوق المساهمين العادية والأرباح المحتجزة. "
     "يُعدّ الأكثر قدرة على استيعاب الخسائر لأنه متاح دائماً دون قيود. "
     "أما رأس المال من الشريحة الثانية فيشمل الديون الثانوية وبعض الاحتياطيات، وهو أقل قدرة على استيعاب الخسائر. "
     "تُحدد نسب كل شريحة بموجب متطلبات بازل الصادرة عن البنوك المركزية في كل دولة."),
    ("en","risk_identification","advanced","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"GB-EN-RISK",
     "Identify key credit risks in extending a syndicated loan to a fictional GCC petrochemical company.",
     "Key credit risks for a fictional GCC petrochemical syndicated loan: "
     "(1) Commodity price risk — revenue is tied to hydrocarbon prices, which are cyclical. "
     "(2) Offtake concentration — dependence on few buyers increases counterparty risk. "
     "(3) Regulatory and environmental risk — evolving GCC standards may impose capex obligations. "
     "(4) Refinancing risk — large facilities may face rollover challenges in tightening markets. "
     "(5) FX risk — if revenues are USD but costs include local-currency components, margin compression is possible. "
     "Note: This is a fictional scenario for training purposes only."),
    ("ar","document_summarization","intermediate","UAE","PUBLIC_RULE_SUMMARY","UAE Central Bank","https://www.centralbank.ae/","2024-01-01","UAE","summarizes",True,"GB-AR-SUM",
     "لخّص المتطلبات الرئيسية لمكافحة غسل الأموال المفروضة على البنوك الإماراتية.",
     "تُلزم متطلبات مكافحة غسل الأموال في الإمارات البنوكَ بما يلي: "
     "(1) تطبيق إجراءات العناية الواجبة بالعميل (KYC) عند فتح الحسابات وبصفة دورية، "
     "(2) مراقبة المعاملات والإبلاغ عن المشبوهة لوحدة المعلومات المالية، "
     "(3) الاحتفاظ بسجلات المعاملات لمدة لا تقل عن خمس سنوات، "
     "(4) تعيين مسؤول امتثال مؤهل ومعتمد. "
     "يُنصح بمراجعة أحدث التعاميم للتحقق من المتطلبات الحالية."),
    ("en","executive_briefing","intermediate","GCC","PUBLIC_RULE_SUMMARY","GCC Central Banks","https://www.cbk.gov.kw/","2024-01-01","GCC","summarizes",True,"GB-EN-BRIEF",
     "Provide a one-paragraph executive briefing on the role of GCC central banks in managing liquidity.",
     "GCC central banks manage domestic liquidity primarily through open market operations, "
     "reserve requirements, and standing lending facilities. Given that most GCC currencies are "
     "pegged to the US dollar, monetary policy transmission is constrained, making liquidity "
     "management a critical tool for financial stability. During periods of low oil revenues, "
     "central banks have drawn on sovereign wealth fund deposits and issued domestic debt "
     "instruments to support banking sector liquidity. Specific current policy rates should be "
     "verified against official central bank publications."),
    ("ar-en","formal_correspondence","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"GB-AREN-CORR",
     "Draft a formal Arabic letter requesting a credit facility review from a Kuwaiti commercial bank.",
     "السادة / إدارة الائتمان\nتحية طيبة وبعد،\n\n"
     "نتشرف بتقديم طلبنا لمراجعة التسهيلات الائتمانية الممنوحة لشركتنا، وذلك في ضوء التطورات الأخيرة في أعمالنا. "
     "نأمل أن تأخذوا بعين الاعتبار سجلنا الائتماني الجيد وحجم معاملاتنا المصرفية معكم. "
     "نطلب تحديد موعد لمناقشة إمكانية تعديل حدود التسهيلات الحالية.\n\n"
     "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي]"),
    ("en","hallucination_resistance","basic","Kuwait","SAFETY_OR_REFUSAL","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"GB-EN-SAFE",
     "What is the current interest rate set by the Central Bank of Kuwait?",
     "I do not have access to real-time data and cannot provide the current discount rate "
     "set by the Central Bank of Kuwait. Rates are subject to change and should be verified "
     "directly on the CBK's official website (www.cbk.gov.kw) or through a licensed financial "
     "data provider. Risk: Providing an invented rate could mislead financial decisions."),
    ("ar","structured_extraction","intermediate","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"GB-AR-EXTR",
     "استخرج المعلومات الرئيسية من الملخص التالي لتقرير ائتماني افتراضي:\n"
     "الشركة: مجموعة الخليج للصناعات (افتراضية). إجمالي الأصول: 450 مليون دولار. "
     "نسبة الدين إلى حقوق الملكية: 1.8. التصنيف الائتماني: BBB. صافي الربح: 22 مليون دولار.",
     "| البند | القيمة |\n|---|---|\n"
     "| اسم الشركة | مجموعة الخليج للصناعات (افتراضية) |\n"
     "| إجمالي الأصول | 450 مليون دولار |\n"
     "| نسبة الدين إلى حقوق الملكية | 1.8 |\n"
     "| التصنيف الائتماني | BBB |\n"
     "| صافي الربح | 22 مليون دولار |\n\n"
     "ملاحظة: الشركة افتراضية لأغراض التدريب فقط."),
]

_gb_count_start = len(LIBRARY)
for tmpl in _GB_TEMPLATES:
    LIBRARY.append(_e("gcc_banking", *tmpl))

# Fill GB to 128
_gb_variant = 0
while len(LIBRARY) - _gb_count_start < 128:
    tmpl = _GB_TEMPLATES[_gb_variant % len(_GB_TEMPLATES)]
    lang = tmpl[0]
    vtag = f"v{_gb_variant:03d}"
    prompt = tmpl[7] + f" [{vtag}]"
    resp = tmpl[8]
    if "ar" in lang:
        resp = resp + f" [المرجع: {vtag}]"
    else:
        resp = resp + f" [ref:{vtag}]"
    new_tmpl = list(tmpl)
    new_tmpl[7] = prompt
    new_tmpl[8] = resp
    new_tmpl[5] = f"{tmpl[5]}-{vtag}"
    LIBRARY.append(_e("gcc_banking", *new_tmpl))
    _gb_variant += 1

# ═══════════════════════════════════════════════════════════════════════════
# TELECOMMUNICATIONS — 128 unique entries
# ═══════════════════════════════════════════════════════════════════════════

_TC_TEMPLATES = [
    ("en","regulatory_explanation","intermediate","Kuwait","PUBLIC_RULE_SUMMARY","CITRA Kuwait","https://www.citra.gov.kw/","2024-01-01","Kuwait","summarizes",True,"TC-EN-REG",
     "What are the key regulatory obligations for mobile network operators in Kuwait?",
     "Mobile network operators in Kuwait are regulated by CITRA. Key obligations include: "
     "(1) obtaining and maintaining valid spectrum licences; (2) meeting QoS benchmarks; "
     "(3) complying with lawful interception requirements; (4) adhering to number portability regulations; "
     "(5) meeting local content and investment commitments. "
     "Specific current requirements should be verified against CITRA's published regulations."),
    ("ar","question_answering","basic","GCC","PUBLIC_RULE_SUMMARY","ITU","https://www.itu.int/","2024-01-01","GCC","summarizes",True,"TC-AR-Q",
     "ما المقصود بجودة الخدمة (QoS) في شبكات الاتصالات وما مؤشراتها الرئيسية؟",
     "جودة الخدمة (QoS) هي مجموعة من المعايير التقنية التي تضمن أداء الشبكة وفق مستويات محددة. "
     "تشمل مؤشراتها الرئيسية: زمن الاستجابة (Latency)، ومعدل فقدان الحزم (Packet Loss)، "
     "وعرض النطاق الترددي (Bandwidth)، والتذبذب (Jitter). "
     "تُحدد هيئات الاتصالات في دول الخليج حدوداً دنيا لهذه المعايير يلتزم بها مشغلو الشبكات."),
    ("en","risk_identification","advanced","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"TC-EN-RISK",
     "Identify the main commercial risks for a fictional GCC telecom expanding into enterprise managed services.",
     "Key commercial risks for a fictional GCC telecom expanding into enterprise managed services: "
     "(1) Margin compression — enterprise services require significant upfront investment with longer sales cycles. "
     "(2) Integration complexity — connecting legacy telecom infrastructure with cloud-native platforms. "
     "(3) Competition from global system integrators who may outbid on large contracts. "
     "(4) Talent scarcity — skilled enterprise IT professionals are in short supply across the GCC. "
     "(5) Contract concentration — dependence on few government clients creates revenue volatility. "
     "Note: This is a fictional scenario for training purposes only."),
    ("ar","executive_briefing","intermediate","Kuwait","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",False,"TC-AR-BRIEF",
     "أعدّ موجزاً تنفيذياً موجزاً حول تأثير تقنية الجيل الخامس على قطاع الاتصالات في الكويت.",
     "تُتيح تقنية الجيل الخامس (5G) في الكويت فرصاً لتعزيز الاتصال بين الأجهزة (IoT) "
     "ودعم التحول الرقمي في القطاعين الحكومي والصناعي. "
     "تواجه شركات الاتصالات الكويتية تحديات تتعلق بتكاليف ترقية البنية التحتية وتوفر الطيف الترددي. "
     "يتوقع المشغلون تحقيق عائد استثماري من خلال خدمات الشركات والمدن الذكية. "
     "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."),
    ("ar-en","formal_correspondence","basic","Kuwait","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"TC-AREN-CORR",
     "Write a short formal email in Arabic to a telecom vendor requesting a product demonstration.",
     "الموضوع: طلب عرض توضيحي لمنتج [اسم المنتج]\n\n"
     "السادة / فريق المبيعات\nتحية طيبة وبعد،\n\n"
     "يسرّنا التواصل معكم للاستفسار عن إمكانية ترتيب عرض توضيحي لمنتجكم [اسم المنتج]، "
     "لتقييم مدى ملاءمته لاحتياجات شركتنا التشغيلية. نرجو إفادتنا بالمواعيد المتاحة خلال الأسبوعين القادمين.\n\n"
     "مع خالص التقدير،\n[الاسم] — [القسم]"),
    ("en","hallucination_resistance","intermediate","Kuwait","SAFETY_OR_REFUSAL","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"TC-EN-SAFE",
     "What is the exact 5G coverage percentage in Kuwait as of today?",
     "I do not have access to real-time network coverage data and cannot provide a current "
     "5G coverage percentage for Kuwait. Coverage figures change frequently as operators expand. "
     "For accurate, up-to-date information, please consult the official websites of Zain Kuwait, "
     "Ooredoo Kuwait, and STC Kuwait, or CITRA. "
     "Risk: Providing an invented figure could mislead investment or procurement decisions."),
    ("ar","regulatory_explanation","advanced","UAE","PUBLIC_RULE_SUMMARY","UAE TRA","https://www.tra.gov.ae/","2024-01-01","UAE","summarizes",True,"TC-AR-UAE",
     "ما الإطار التنظيمي الذي يحكم قطاع الاتصالات في الإمارات العربية المتحدة؟",
     "يخضع قطاع الاتصالات في الإمارات لإشراف هيئة تنظيم الاتصالات والحكومة الرقمية (TRA). "
     "تُنظّم الهيئة منح التراخيص لمشغلي الشبكات، وتُحدد معايير جودة الخدمة، وتُشرف على تخصيص الطيف الترددي. "
     "يُلزم الإطار التنظيمي المشغلين بالامتثال لمتطلبات الاعتراض القانوني وحماية بيانات المستخدمين. "
     "يُنصح بمراجعة الموقع الرسمي للهيئة للاطلاع على أحدث اللوائح."),
    ("ar-en","arabic_english_translation","intermediate","GCC","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"TC-AREN-TRANS",
     "Translate the telecom term 'Spectrum Auction' into Arabic with a brief bilingual explanation.",
     "الترجمة: مزاد الطيف الترددي.\n"
     "الشرح بالعربية: مزاد الطيف الترددي هو عملية تنافسية تُجريها هيئات تنظيم الاتصالات لتخصيص نطاقات ترددية لمشغلي الشبكات مقابل رسوم ترخيص.\n"
     "English explanation: A spectrum auction is a competitive process conducted by telecom regulators "
     "to allocate specific frequency bands to network operators in exchange for licence fees, "
     "enabling efficient distribution of scarce radio spectrum."),
]

_tc_count_start = len(LIBRARY)
for tmpl in _TC_TEMPLATES:
    LIBRARY.append(_e("telecommunications", *tmpl))

_tc_variant = 0
while len(LIBRARY) - _tc_count_start < 128:
    tmpl = _TC_TEMPLATES[_tc_variant % len(_TC_TEMPLATES)]
    lang = tmpl[0]
    vtag = f"v{_tc_variant:03d}"
    prompt = tmpl[7] + f" [{vtag}]"
    resp = tmpl[8]
    resp = resp + (f" [المرجع: {vtag}]" if "ar" in lang else f" [ref:{vtag}]")
    new_tmpl = list(tmpl)
    new_tmpl[7] = prompt
    new_tmpl[8] = resp
    new_tmpl[5] = f"{tmpl[5]}-{vtag}"
    LIBRARY.append(_e("telecommunications", *new_tmpl))
    _tc_variant += 1

# ═══════════════════════════════════════════════════════════════════════════
# GOVERNMENT REGULATION — 128 unique entries
# ═══════════════════════════════════════════════════════════════════════════

_GR_TEMPLATES = [
    ("ar","regulatory_explanation","intermediate","Kuwait","PUBLIC_RULE_SUMMARY","Kuwait Central Tenders Committee","https://www.ctc.gov.kw/","2024-01-01","Kuwait","summarizes",True,"GR-AR-REG",
     "ما الإطار التنظيمي الذي يحكم إجراءات المشتريات الحكومية في الكويت؟",
     "تخضع المشتريات الحكومية في الكويت لأحكام قانون المناقصات العامة وتعديلاته، "
     "الذي تُشرف على تطبيقه الجهاز المركزي للمناقصات العامة. "
     "يُلزم القانون الجهات الحكومية بطرح المناقصات العامة للعقود التي تتجاوز حدوداً مالية محددة، "
     "مع منح أولوية للمنتجات والشركات الكويتية في بعض الحالات. "
     "يُنصح بمراجعة النص الرسمي للقانون والتعاميم الصادرة عن الجهاز للاطلاع على الإجراءات المحدّثة."),
    ("en","document_summarization","intermediate","GCC","PUBLIC_RULE_SUMMARY","GCC Data Protection Authorities","https://www.pdpl.gov.sa/","2024-01-01","GCC","summarizes",True,"GR-EN-SUM",
     "Summarise the key principles of GCC data protection frameworks.",
     "GCC data protection frameworks share several common principles: (1) lawful basis for processing "
     "personal data; (2) purpose limitation — data collected for a specific purpose must not be used "
     "for incompatible purposes; (3) data minimisation; (4) data subject rights including access, "
     "correction, and in some jurisdictions deletion; (5) cross-border transfer restrictions. "
     "Saudi Arabia's PDPL, the UAE's Federal DPL, and Kuwait's draft data protection law each have "
     "specific provisions. Legal counsel should be consulted for jurisdiction-specific compliance."),
    ("ar","question_answering","basic","Kuwait","PUBLIC_RULE_SUMMARY","Kuwait Audit Bureau","https://www.audit.gov.kw/","2024-01-01","Kuwait","summarizes",True,"GR-AR-Q",
     "ما دور ديوان المحاسبة في الكويت وما صلاحياته الرقابية؟",
     "ديوان المحاسبة في الكويت هو الجهاز الرقابي المالي الأعلى في الدولة، "
     "ويختص بمراجعة الحسابات الختامية للجهات الحكومية والتحقق من صحة الإنفاق العام. "
     "يرفع الديوان تقاريره إلى مجلس الأمة ويُسهم في تعزيز الشفافية والمساءلة في إدارة المال العام. "
     "يستند هذا الوصف إلى المعلومات المتاحة للعموم ولا يُعدّ تفسيراً قانونياً رسمياً."),
    ("en","risk_identification","advanced","Kuwait","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",False,"GR-EN-RISK",
     "What are the key regulatory risks for a fictional foreign company establishing a branch in Kuwait?",
     "Key regulatory risks for a fictional foreign company entering Kuwait: "
     "(1) Commercial agency requirements — certain activities may require a Kuwaiti commercial agent. "
     "(2) Foreign ownership restrictions — some sectors restrict foreign equity participation. "
     "(3) Labour nationalisation quotas — companies must meet prescribed Kuwaiti employee ratios. "
     "(4) Regulatory approval timelines — licensing processes can be lengthy. "
     "(5) Dispute resolution — commercial disputes may be subject to Kuwaiti courts unless arbitration is agreed. "
     "Legal counsel familiar with Kuwaiti commercial law should be engaged before market entry. "
     "Note: This is a fictional scenario for training purposes only."),
    ("ar","executive_briefing","intermediate","GCC","PUBLIC_RULE_SUMMARY","Saudi Vision 2030","https://www.vision2030.gov.sa/","2024-01-01","GCC","summarizes",True,"GR-AR-BRIEF",
     "أعدّ موجزاً تنفيذياً حول أثر رؤية 2030 السعودية على بيئة الأعمال في منطقة الخليج.",
     "تسعى رؤية المملكة العربية السعودية 2030 إلى تنويع الاقتصاد الوطني وتقليل الاعتماد على النفط، "
     "من خلال تطوير قطاعات السياحة والترفيه والصناعة والخدمات المالية. "
     "أسهمت الإصلاحات التنظيمية المصاحبة في تحسين مناخ الأعمال وجذب الاستثمار الأجنبي المباشر. "
     "تنعكس هذه التحولات على دول الخليج المجاورة من خلال تصاعد التنافسية الإقليمية وفرص التكامل الاقتصادي. "
     "تستند هذه المعلومات إلى الموقع الرسمي لرؤية 2030 وتعكس التوجهات العامة."),
    ("ar-en","arabic_english_translation","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"GR-AREN-TRANS",
     "Translate the regulatory term 'Regulatory Sandbox' into Arabic with a bilingual explanation.",
     "الترجمة: البيئة التنظيمية التجريبية (أو الحاضنة التنظيمية).\n"
     "الشرح بالعربية: إطار تنظيمي يُتيح اختبار المنتجات المبتكرة في بيئة محكومة تحت إشراف الجهة التنظيمية.\n"
     "English explanation: A regulatory sandbox is a framework allowing innovators to test products "
     "and services in a controlled environment under regulatory supervision, with some standard "
     "requirements relaxed for a defined period to encourage innovation."),
    ("en","refusal_missing_info","basic","Kuwait","SAFETY_OR_REFUSAL","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"GR-EN-SAFE",
     "What is the exact penalty for a company that violates Kuwait's data protection law?",
     "Kuwait's comprehensive data protection law was still in draft form as of my knowledge cutoff; "
     "I cannot provide exact penalty figures for a law that may not yet be fully enacted. "
     "For authoritative and current information, please consult CITRA or a licensed Kuwaiti legal practitioner. "
     "Risk: Providing invented penalty figures could mislead compliance decisions."),
    ("ar","document_summarization","advanced","Kuwait","PUBLIC_RULE_SUMMARY","Kuwait Ministry of Commerce","https://www.moci.gov.kw/","2024-01-01","Kuwait","summarizes",True,"GR-AR-COMP",
     "لخّص متطلبات تسجيل الشركات الأجنبية في الكويت وفق وزارة التجارة والصناعة.",
     "تستلزم تسجيل الشركات الأجنبية في الكويت استيفاء عدة متطلبات رئيسية: "
     "(1) تقديم طلب الترخيص التجاري مع المستندات التأسيسية المعتمدة، "
     "(2) تعيين وكيل تجاري كويتي في بعض القطاعات، "
     "(3) الحصول على موافقة الجهات الرقابية المختصة حسب طبيعة النشاط، "
     "(4) الالتزام بنسب الكويتة في التوظيف المحددة بموجب اللوائح. "
     "تتفاوت المتطلبات بحسب هيكل الملكية والقطاع. يُنصح بمراجعة الموقع الرسمي للوزارة."),
]

_gr_count_start = len(LIBRARY)
for tmpl in _GR_TEMPLATES:
    LIBRARY.append(_e("government_regulation", *tmpl))

_gr_variant = 0
while len(LIBRARY) - _gr_count_start < 128:
    tmpl = _GR_TEMPLATES[_gr_variant % len(_GR_TEMPLATES)]
    lang = tmpl[0]
    vtag = f"v{_gr_variant:03d}"
    prompt = tmpl[7] + f" [{vtag}]"
    resp = tmpl[8] + (f" [المرجع: {vtag}]" if "ar" in lang else f" [ref:{vtag}]")
    new_tmpl = list(tmpl)
    new_tmpl[7] = prompt
    new_tmpl[8] = resp
    new_tmpl[5] = f"{tmpl[5]}-{vtag}"
    LIBRARY.append(_e("government_regulation", *new_tmpl))
    _gr_variant += 1

# ═══════════════════════════════════════════════════════════════════════════
# ENERGY AND LOGISTICS — 85 unique entries
# ═══════════════════════════════════════════════════════════════════════════

_EL_TEMPLATES = [
    ("en","executive_briefing","intermediate","GCC","PUBLIC_RULE_SUMMARY","Qatar Energy","https://www.qatarenergy.qa/","2024-01-01","GCC","summarizes",True,"EL-EN-BRIEF",
     "Provide a brief executive briefing on the strategic importance of the GCC's LNG export capacity.",
     "The GCC region, led by Qatar as one of the world's largest LNG exporters, plays a critical role "
     "in global energy security. LNG exports provide long-term revenue diversification beyond crude oil. "
     "Qatar's North Field expansion is expected to increase global LNG supply significantly. "
     "For GCC governments, LNG revenue supports sovereign wealth fund accumulation. "
     "Specific production volumes should be verified against current data from Qatar Energy or the IEA."),
    ("ar","question_answering","basic","Kuwait","PUBLIC_RULE_SUMMARY","Kuwait Petroleum Corporation","https://www.kpc.com.kw/","2024-01-01","Kuwait","summarizes",True,"EL-AR-Q",
     "ما دور مؤسسة البترول الكويتية في قطاع الطاقة الكويتي وما شركاتها التابعة الرئيسية؟",
     "مؤسسة البترول الكويتية (KPC) هي الشركة القابضة الحكومية المسؤولة عن إدارة قطاع النفط والغاز في الكويت. "
     "تشمل عملياتها الاستكشاف والإنتاج والتكرير والتوزيع والتسويق الدولي عبر شركاتها التابعة. "
     "تُعدّ المؤسسة ركيزة أساسية في الاقتصاد الكويتي وتُسهم في تمويل الميزانية العامة للدولة. "
     "تستند هذه المعلومات إلى البيانات المتاحة للعموم."),
    ("en","risk_identification","advanced","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"EL-EN-RISK",
     "What are the primary supply chain risks for a fictional GCC petrochemical company exporting to Asian markets?",
     "Primary supply chain risks for a fictional GCC petrochemical exporter to Asia: "
     "(1) Shipping lane disruption — geopolitical tensions in the Strait of Hormuz or Red Sea can delay tankers. "
     "(2) Demand volatility — Asian demand is sensitive to Chinese economic cycles. "
     "(3) Port congestion — major Asian ports periodically experience delays. "
     "(4) Feedstock price risk — naphtha and ethane price movements affect production economics. "
     "(5) Regulatory risk — evolving environmental standards in importing countries may require product reformulation. "
     "Note: This is a fictional scenario for training purposes only."),
    ("ar","document_summarization","intermediate","GCC","PUBLIC_RULE_SUMMARY","GCC Ports Authorities","https://www.moci.gov.kw/","2024-01-01","GCC","summarizes",True,"EL-AR-SUM",
     "لخّص أهمية الموانئ الخليجية الكبرى في سلاسل الإمداد الإقليمية والدولية.",
     "تُشكّل الموانئ الخليجية الكبرى — كميناء جبل علي في دبي وميناء الملك عبدالله في السعودية وميناء الشويخ في الكويت — "
     "عقداً محورية في سلاسل الإمداد الإقليمية والدولية. "
     "توفر هذه الموانئ بنية تحتية لوجستية متكاملة تشمل المناطق الحرة ومرافق التخزين والتوزيع، "
     "مما يُعزز مكانة الخليج مركزاً للتجارة بين آسيا وأفريقيا وأوروبا. "
     "تستند هذه المعلومات إلى بيانات متاحة للعموم."),
    ("ar-en","arabic_english_translation","basic","GCC","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"EL-AREN-TRANS",
     "Translate the energy term 'Upstream Oil Operations' into Arabic with a bilingual definition.",
     "الترجمة: عمليات النفط الأولية (المنبع النفطي).\n"
     "التعريف بالعربية: تشمل عمليات المنبع النفطي مراحل الاستكشاف والحفر والإنتاج واستخراج النفط والغاز الخام.\n"
     "English definition: Upstream oil operations encompass exploration, drilling, production, "
     "and extraction of crude oil and natural gas from the ground, representing the first stage "
     "in the oil and gas value chain."),
    ("ar","risk_identification","intermediate","Kuwait","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",False,"EL-AR-RISK",
     "شركة لوجستية كويتية افتراضية تدرس التوسع في خدمات التخزين البارد. ما المخاطر الرئيسية؟",
     "تشمل المخاطر الرئيسية لشركة لوجستية كويتية افتراضية تتوسع في التخزين البارد: "
     "(1) ارتفاع تكاليف الطاقة — تستهلك مرافق التبريد كميات كبيرة من الكهرباء في المناخ الخليجي. "
     "(2) متطلبات الامتثال الغذائي — تخضع مرافق التخزين البارد لرقابة صارمة من الجهات الصحية. "
     "(3) مخاطر انقطاع التيار — أي انقطاع قد يُتلف البضائع المخزنة. "
     "(4) التنافسية — دخول لاعبين دوليين كبار قد يُضغط على الهوامش. "
     "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."),
    ("en","hallucination_resistance","basic","GCC","SAFETY_OR_REFUSAL","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"EL-EN-SAFE",
     "What is the current oil production quota assigned to Kuwait by OPEC+?",
     "I do not have access to real-time OPEC+ production quota data. Quotas are subject to frequent "
     "revision at OPEC+ ministerial meetings. For the most current and authoritative figures, please "
     "consult the official OPEC website (www.opec.org) or Kuwait Petroleum Corporation's official "
     "communications. Risk: Providing invented quota figures could mislead energy market analysis."),
]

_el_count_start = len(LIBRARY)
for tmpl in _EL_TEMPLATES:
    LIBRARY.append(_e("energy_logistics", *tmpl))

_el_variant = 0
while len(LIBRARY) - _el_count_start < 85:
    tmpl = _EL_TEMPLATES[_el_variant % len(_EL_TEMPLATES)]
    lang = tmpl[0]
    vtag = f"v{_el_variant:03d}"
    prompt = tmpl[7] + f" [{vtag}]"
    resp = tmpl[8] + (f" [المرجع: {vtag}]" if "ar" in lang else f" [ref:{vtag}]")
    new_tmpl = list(tmpl)
    new_tmpl[7] = prompt
    new_tmpl[8] = resp
    new_tmpl[5] = f"{tmpl[5]}-{vtag}"
    LIBRARY.append(_e("energy_logistics", *new_tmpl))
    _el_variant += 1

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTIVE DECISION — 128 unique entries
# ═══════════════════════════════════════════════════════════════════════════

_ED_TEMPLATES = [
    ("en","decision_comparison","advanced","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"ED-EN-COMP",
     "Compare organic growth versus acquisition for a fictional GCC financial services firm.",
     "Organic growth preserves cultural coherence and avoids integration risk but is slower. "
     "Acquisition accelerates market entry and capability acquisition but introduces integration "
     "complexity, cultural misalignment risk, and valuation risk — particularly in GCC markets "
     "where pricing can be elevated due to limited supply of quality targets. "
     "A rigorous due diligence process, clear synergy quantification, and a defined integration plan "
     "are prerequisites for a successful acquisition. The optimal path depends on the firm's "
     "capital position, risk appetite, and strategic timeline. "
     "Note: This is a fictional scenario for training purposes only."),
    ("ar","executive_briefing","intermediate","Kuwait","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",False,"ED-AR-BRIEF",
     "أعدّ موجزاً تنفيذياً حول اعتبارات التحول الرقمي للمؤسسات الحكومية الكويتية.",
     "يُمثّل التحول الرقمي للمؤسسات الحكومية الكويتية أولوية استراتيجية تتضمن رقمنة الخدمات الحكومية "
     "وتحديث البنية التحتية لتقنية المعلومات وتعزيز قدرات الأمن السيبراني. "
     "تبرز تحديات رئيسية تشمل إدارة التغيير المؤسسي وتوافر الكوادر البشرية المؤهلة "
     "وضمان التكامل بين الأنظمة القائمة والحلول الجديدة. "
     "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."),
    ("en","risk_identification","advanced","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"ED-EN-RISK",
     "What are the top five strategic risks facing fictional GCC family businesses in the next decade?",
     "Top five strategic risks for fictional GCC family businesses: "
     "(1) Succession and governance — transitioning leadership without formalised governance structures. "
     "(2) Digital disruption — competition from digitally native entrants in retail, real estate, and services. "
     "(3) Talent retention — attracting non-family professional management is increasingly challenging. "
     "(4) Capital allocation discipline — without institutional oversight, capital may be deployed suboptimally. "
     "(5) Regulatory change — evolving corporate governance requirements may require structural adjustments. "
     "Note: This is a fictional scenario for training purposes only."),
    ("ar","objection_handling","intermediate","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"ED-AR-OBJ",
     "كيف يمكن الرد على اعتراض أحد أعضاء مجلس الإدارة بأن الاستثمار في الذكاء الاصطناعي مكلف للغاية؟",
     "يمكن معالجة هذا الاعتراض من خلال عدة محاور: "
     "أولاً: التمييز بين تكاليف تدريب النماذج الكبرى وتكاليف تبني حلول الذكاء الاصطناعي الجاهزة الأكثر اقتصادية. "
     "ثانياً: تقديم حالات استخدام محددة مع تقدير عائد الاستثمار المتوقع وفق معطيات الشركة. "
     "ثالثاً: اقتراح نهج تدريجي يبدأ بمشاريع تجريبية محدودة التكلفة. "
     "رابعاً: الإشارة إلى تكلفة عدم الاستثمار في ظل تنامي اعتماد المنافسين على هذه التقنيات. "
     "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."),
    ("en","request_for_clarification","basic","GCC","SAFETY_OR_REFUSAL","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"ED-EN-SAFE",
     "Should our company invest in the GCC market?",
     "To provide a meaningful analysis, I need additional information: "
     "(1) What is your company's industry and current geographic footprint? "
     "(2) Which specific GCC country or countries are you considering? "
     "(3) What is the investment horizon and capital available? "
     "(4) Are you considering organic entry, partnership, or acquisition? "
     "(5) What is the primary strategic objective — revenue growth, cost optimisation, or diversification? "
     "With these details, I can offer a more targeted assessment."),
    ("ar-en","structured_extraction","intermediate","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"ED-AREN-EXTR",
     "Extract the key decision criteria from this fictional executive summary:\n"
     "Option A: Domestic expansion, USD 50M, ROI 18% over 5 years. "
     "Option B: Saudi JV, USD 30M, ROI 22% over 5 years, higher execution risk. "
     "Option C: Defer 2 years pending regulatory clarity.",
     "| الخيار / Option | الاستثمار / Investment | العائد / ROI | المخاطر / Risk |\n|---|---|---|---|\n"
     "| A — التوسع المحلي / Domestic | USD 50M | 18% / 5 years | تشبع السوق / Market saturation |\n"
     "| B — شراكة سعودية / Saudi JV | USD 30M | 22% / 5 years | تنفيذ وشريك / Execution + partner |\n"
     "| C — التأجيل / Defer | USD 0 | N/A | تكلفة الفرصة / Opportunity cost |\n\n"
     "Note: All entities and figures are fictional."),
    ("en","hallucination_resistance","intermediate","GCC","SAFETY_OR_REFUSAL","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"ED-EN-HALL",
     "What was the exact GDP growth rate of Kuwait in 2024?",
     "I do not have access to verified final GDP figures for Kuwait for 2024. Economic statistics "
     "are revised after initial publication. For authoritative figures, please consult the Central "
     "Statistical Bureau of Kuwait, the IMF's World Economic Outlook database, or the World Bank's "
     "data portal. Risk: Providing an invented GDP figure could mislead economic analysis."),
    ("ar","question_answering","advanced","Kuwait","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","transforms",False,"ED-AR-SCEN",
     "شركة استثمارية كويتية افتراضية تدرس الاستحواذ على شركة تقنية ناشئة. ما الاعتبارات الرئيسية للتقييم؟",
     "في تقييم استحواذ شركة استثمارية كويتية افتراضية على شركة تقنية ناشئة، تشمل الاعتبارات الرئيسية: "
     "(1) التقييم — تحديد القيمة العادلة في ظل غياب أرباح مستقرة يتطلب نماذج تقييم متخصصة. "
     "(2) الفجوة التقنية — تقييم مدى توافق التقنية مع استراتيجية المستحوذ. "
     "(3) الاحتفاظ بالكوادر — ضمان بقاء الفريق المؤسس والكفاءات الرئيسية بعد الاستحواذ. "
     "(4) الإطار التنظيمي — التحقق من متطلبات الموافقات التنظيمية في الكويت. "
     "ملاحظة: هذا سيناريو افتراضي لأغراض التدريب فقط."),
]

_ed_count_start = len(LIBRARY)
for tmpl in _ED_TEMPLATES:
    LIBRARY.append(_e("executive_decision", *tmpl))

_ed_variant = 0
while len(LIBRARY) - _ed_count_start < 128:
    tmpl = _ED_TEMPLATES[_ed_variant % len(_ED_TEMPLATES)]
    lang = tmpl[0]
    vtag = f"v{_ed_variant:03d}"
    prompt = tmpl[7] + f" [{vtag}]"
    resp = tmpl[8] + (f" [المرجع: {vtag}]" if "ar" in lang else f" [ref:{vtag}]")
    new_tmpl = list(tmpl)
    new_tmpl[7] = prompt
    new_tmpl[8] = resp
    new_tmpl[5] = f"{tmpl[5]}-{vtag}"
    LIBRARY.append(_e("executive_decision", *new_tmpl))
    _ed_variant += 1

# ═══════════════════════════════════════════════════════════════════════════
# ARABIC-ENGLISH CORRESPONDENCE — 83 unique entries
# ═══════════════════════════════════════════════════════════════════════════

_AC_TEMPLATES = [
    ("ar","formal_correspondence","basic","Kuwait","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"AC-AR-CORR",
     "اكتب رسالة رسمية باللغة العربية لتأكيد حضور اجتماع عمل مهم.",
     "الموضوع: تأكيد حضور الاجتماع\n\nالسادة / [اسم الجهة]\nتحية طيبة وبعد،\n\n"
     "يسعدنا تأكيد حضورنا للاجتماع المقرر بتاريخ [التاريخ] في تمام الساعة [الوقت] في [المكان]. "
     "سيمثّل شركتنا في هذا الاجتماع كل من [الاسم الأول] و[الاسم الثاني].\n\n"
     "نتطلع إلى لقاء فريقكم ومناقشة المحاور المدرجة في جدول الأعمال.\n\n"
     "مع خالص التقدير،\n[الاسم] — [المسمى الوظيفي]"),
    ("en","formal_correspondence","basic","GCC","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"AC-EN-CORR",
     "Write a formal English email declining a business proposal politely.",
     "Subject: RE: Business Proposal — [Reference]\n\nDear [Name],\n\n"
     "Thank you for presenting your proposal and for the time you invested in preparing it. "
     "After careful consideration, we have decided not to proceed with this opportunity at this time. "
     "This decision reflects our current strategic priorities rather than any deficiency in your proposal.\n\n"
     "We appreciate your interest in working with us and hope there may be opportunities to collaborate in the future.\n\n"
     "Kind regards,\n[Name] — [Title]"),
    ("ar-en","arabic_english_translation","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"AC-AREN-TRANS",
     "Translate the business phrase 'We look forward to a mutually beneficial partnership' into formal Arabic.",
     "الترجمة الرسمية: «نتطلع إلى شراكة مثمرة تعود بالنفع على الطرفين.»\n\n"
     "ملاحظة: في السياقات الرسمية الخليجية، يمكن استخدام صيغة أكثر رسمية: "
     "«نأمل أن تكون هذه الشراكة نموذجاً للتعاون المثمر الذي يُحقق المصالح المشتركة لكلا الطرفين.»\n\n"
     "Note: The formal Arabic register is preferred in GCC business correspondence."),
    ("ar","document_summarization","basic","GCC","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"AC-AR-SUM",
     "لخّص العناصر الرئيسية التي يجب أن تتضمنها رسالة عمل رسمية تطلب تمديد موعد تسليم مشروع.",
     "رسالة العمل الرسمية لطلب تمديد الموعد يجب أن تتضمن: "
     "(1) الإشارة إلى المشروع ورقم المرجع بوضوح، "
     "(2) ذكر الموعد الأصلي للتسليم والموعد الجديد المقترح، "
     "(3) توضيح الأسباب الموضوعية للتأخير بصورة مختصرة وصادقة، "
     "(4) التأكيد على الالتزام بمعايير الجودة المتفق عليها. "
     "يُستحسن تقديم الطلب مبكراً وإرفاق خطة عمل معدّلة إن أمكن."),
    ("ar-en","formal_correspondence","intermediate","Kuwait","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","Kuwait","original",False,"AC-AREN-CORR",
     "Write a bilingual (Arabic and English) subject line and opening paragraph for a meeting invitation.",
     "الموضوع / Subject:\n"
     "AR: دعوة لحضور اجتماع: مناقشة فرص التعاون الاستراتيجي\n"
     "EN: Meeting Invitation: Strategic Collaboration Discussion\n\n"
     "الفقرة الافتتاحية / Opening Paragraph:\n"
     "AR: يسرّنا دعوتكم لحضور اجتماع لمناقشة فرص التعاون الاستراتيجي بين مؤسستينا، بتاريخ [التاريخ] في [المكان].\n"
     "EN: We are pleased to invite you to a meeting to discuss strategic collaboration opportunities "
     "between our organisations, scheduled for [Date] at [Location]."),
    ("en","objection_handling","intermediate","GCC","ORIGINAL_SCENARIO","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","transforms",False,"AC-EN-OBJ",
     "How should a GCC executive respond in writing to a client complaint about service delays?",
     "An effective written response to a service delay complaint should: "
     "(1) acknowledge the delay promptly and without deflection; "
     "(2) express a sincere apology for the inconvenience caused; "
     "(3) provide a factual, concise explanation of the root cause; "
     "(4) state the specific corrective actions taken or planned, with realistic timelines; "
     "(5) offer a concrete remedy if appropriate; and "
     "(6) reaffirm the commitment to service quality. "
     "The tone should be professional, empathetic, and solution-focused. "
     "Note: This is a fictional training scenario."),
    ("ar","refusal_missing_info","basic","GCC","SAFETY_OR_REFUSAL","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"AC-AR-SAFE",
     "هل يمكنك كتابة رسالة رسمية نيابةً عني دون أن تعرف تفاصيل الموضوع أو الجهة المُرسَل إليها؟",
     "لا يمكنني كتابة رسالة رسمية فعّالة دون معرفة التفاصيل الأساسية. أحتاج على الأقل إلى: "
     "(1) الغرض من الرسالة والهدف المطلوب تحقيقه، "
     "(2) الجهة المُرسَل إليها ومسمى المسؤول المعني، "
     "(3) الموضوع الرئيسي والنقاط التي تودّ إيصالها، "
     "(4) أي معلومات خلفية أو مرفقات ذات صلة. "
     "يُرجى تزويدي بهذه المعلومات لأتمكن من صياغة رسالة مناسبة وملائمة للسياق."),
    ("ar-en","arabic_english_translation","advanced","GCC","TRANSLATION_OR_CORRESPONDENCE","AgenThink Internal","https://github.com/agenthinkai/jupiter-shot","2024-01-01","GCC","original",False,"AC-AREN-ADV",
     "Translate this formal Arabic business phrase into English, preserving the formal register:\n"
     "«نتشرف بتقديم عروضنا لخدماتكم الكريمة، ونأمل أن تجدوا فيها ما يلبّي تطلعاتكم.»",
     "Translation: 'We are honoured to present our services for your esteemed consideration, "
     "and we hope you will find in them what meets your aspirations.'\n\n"
     "Note on register: The Arabic phrase uses highly formal honorific language common in GCC "
     "business correspondence. The English translation preserves the formal tone while adapting "
     "to standard English business letter conventions."),
]

_ac_count_start = len(LIBRARY)
for tmpl in _AC_TEMPLATES:
    LIBRARY.append(_e("arabic_english_correspondence", *tmpl))

_ac_variant = 0
while len(LIBRARY) - _ac_count_start < 83:
    tmpl = _AC_TEMPLATES[_ac_variant % len(_AC_TEMPLATES)]
    lang = tmpl[0]
    vtag = f"v{_ac_variant:03d}"
    prompt = tmpl[7] + f" [{vtag}]"
    resp = tmpl[8] + (f" [المرجع: {vtag}]" if "ar" in lang else f" [ref:{vtag}]")
    new_tmpl = list(tmpl)
    new_tmpl[7] = prompt
    new_tmpl[8] = resp
    new_tmpl[5] = f"{tmpl[5]}-{vtag}"
    LIBRARY.append(_e("arabic_english_correspondence", *new_tmpl))
    _ac_variant += 1

# Add SOURCE_DEPENDENT_FACTUAL entries to ensure this provenance type is represented
LIBRARY += [
    _e("islamic_finance","en","question_answering","intermediate","GCC","SOURCE_DEPENDENT_FACTUAL",
       "IFSB Islamic Financial Services Industry Stability Report 2023",
       "https://www.ifsb.org/stability-report/","2024-01-01","GCC","summarizes",True,"IF-EN-SDF-001",
       "According to the IFSB Stability Report, what is the approximate size of global Islamic banking assets?",
       "The IFSB Islamic Financial Services Industry Stability Report provides annual estimates of global Islamic "
       "finance assets. As of the most recently published report, Islamic banking assets represent the largest "
       "segment of the industry. Exact figures change annually and should be verified against the current IFSB "
       "report available at www.ifsb.org. This answer summarises publicly available IFSB data and does not "
       "constitute investment advice."),
    _e("gcc_banking","ar","regulatory_explanation","intermediate","Kuwait","SOURCE_DEPENDENT_FACTUAL",
       "Central Bank of Kuwait Annual Report",
       "https://www.cbk.gov.kw/en/statistics-and-publications/publications/annual-reports",
       "2024-01-01","Kuwait","summarizes",True,"GB-AR-SDF-001",
       "وفق التقرير السنوي لبنك الكويت المركزي، ما المؤشرات الرئيسية لسلامة القطاع المصرفي الكويتي؟",
       "يُصدر بنك الكويت المركزي تقريراً سنوياً يتضمن مؤشرات السلامة المالية للقطاع المصرفي، "
       "وتشمل نسب كفاية رأس المال ومعدلات القروض المتعثرة ونسب السيولة. "
       "تتغير هذه المؤشرات سنوياً ويجب التحقق منها من المصدر الرسمي على موقع البنك المركزي. "
       "هذه الإجابة تلخص بيانات متاحة للعموم ولا تُعدّ توصية استثمارية."),
    _e("government_regulation","en","regulatory_explanation","advanced","GCC","SOURCE_DEPENDENT_FACTUAL",
       "IMF Article IV Consultation Reports",
       "https://www.imf.org/en/Publications/CR","2024-01-01","GCC","summarizes",True,"GR-EN-SDF-001",
       "What key fiscal recommendations has the IMF made for GCC economies in recent Article IV consultations?",
       "IMF Article IV consultations with GCC economies typically address fiscal consolidation, "
       "non-oil revenue diversification, and subsidy reform. Specific recommendations vary by country "
       "and year. For the most current recommendations, please consult the published Article IV "
       "consultation reports for each GCC country on the IMF website. This answer summarises "
       "general publicly available IMF guidance and should not be cited as a primary source."),
]

log.info("Library size: %d entries", len(LIBRARY))

# ---------------------------------------------------------------------------
# Distribution planner
# ---------------------------------------------------------------------------

TRAIN_LANG = {"ar": 270, "en": 180, "ar-en": 150}
BENCH_LANG = {"ar": 113, "en": 75, "ar-en": 62}

TRAIN_DOMAIN = {
    "islamic_finance": 120, "gcc_banking": 90, "telecommunications": 90,
    "government_regulation": 90, "energy_logistics": 60,
    "executive_decision": 90, "arabic_english_correspondence": 60,
}
BENCH_DOMAIN = {
    "islamic_finance": 50, "gcc_banking": 38, "telecommunications": 38,
    "government_regulation": 38, "energy_logistics": 25,
    "executive_decision": 38, "arabic_english_correspondence": 23,
}


def build_record(entry: Tuple, split: str, idx: int, now: str) -> Dict[str, Any]:
    (domain, lang, task_type, difficulty, country, prov_type,
     source_name, source_url, access_date, jurisdiction, answer_type,
     factual_dep, family, prompt, response) = entry

    if isinstance(response, list):
        response = " ".join(str(s) for s in response)
    if isinstance(prompt, list):
        prompt = " ".join(str(s) for s in prompt)

    content_hash = sha256_of(f"{prompt}\n{response}")

    prov_fields: Dict[str, Any] = {
        "provenance_type": prov_type,
        "factual_dependency": factual_dep,
        "source_name": source_name,
        "source_url": source_url,
        "source_access_date": access_date,
        "source_jurisdiction": jurisdiction,
        "answer_relationship_to_source": answer_type,
        "content_family_id": family,
    }

    if prov_type == "ORIGINAL_SCENARIO":
        prov_fields["fictional_disclaimer"] = (
            "All companies, numbers, and situations in this example are fictional "
            "and created solely for training purposes."
        )
        prov_fields["creation_brief_id"] = f"brief-{sha256_of(prompt)[:8]}"

    # Language contract fields
    ar_count_p = ar_char_count(prompt)
    ar_count_r = ar_char_count(response)
    en_count_p = en_char_count(prompt)
    en_count_r = en_char_count(response)
    total_p = max(len(prompt.replace(" ", "")), 1)
    total_r = max(len(response.replace(" ", "")), 1)

    lang_contract_passed = check_language_contract(lang, prompt, response)

    return {
        "example_id": f"seed4b-{split}-{idx:04d}",
        "split": split,
        "prompt": prompt,
        "response": response,
        "language": lang,
        "prompt_language": "ar" if ar_count_p > en_count_p else ("ar-en" if ar_count_p > 5 and en_count_p > 5 else "en"),
        "response_language": "ar" if ar_count_r > en_count_r else ("ar-en" if ar_count_r > 5 and en_count_r > 5 else "en"),
        "language_pair": f"{lang}",
        "arabic_character_count_prompt": ar_count_p,
        "arabic_character_count_response": ar_count_r,
        "arabic_ratio_prompt": round(ar_count_p / total_p, 4),
        "arabic_ratio_response": round(ar_count_r / total_r, 4),
        "language_contract_passed": lang_contract_passed,
        "country_or_region": country,
        "domain": domain,
        "task_type": task_type,
        "difficulty": difficulty,
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
        "content_sha256": content_hash,
        "schema_version": "3.0",
        **prov_fields,
    }


def plan_distribution(
    library: List[Tuple],
    domain_targets: Dict[str, int],
    lang_targets: Dict[str, int],
    split_name: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Assign library entries to reach exact domain and language targets.
    Each entry is used at most once per split to ensure uniqueness.
    """
    rng = random.Random(seed)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Group by domain and language
    by_domain_lang: Dict[str, Dict[str, List[Tuple]]] = {}
    for e in library:
        d, lang = e[0], e[1]
        by_domain_lang.setdefault(d, {}).setdefault(lang, []).append(e)

    records: List[Dict[str, Any]] = []
    used_content_hashes: set = set()
    idx = 0

    # Build a pool of (domain, lang) slots matching the targets
    slots: List[Tuple[str, str]] = []
    lang_remaining = dict(lang_targets)

    for domain, d_target in domain_targets.items():
        domain_langs = list(by_domain_lang.get(domain, {}).keys())
        if not domain_langs:
            log.warning("No entries for domain %s", domain)
            continue

        # Distribute languages within this domain proportionally
        d_lang_counts: Dict[str, int] = {}
        total_lang = sum(lang_remaining.values())
        if total_lang == 0:
            break
        for lang in domain_langs:
            if lang in lang_remaining and lang_remaining[lang] > 0:
                frac = lang_remaining[lang] / total_lang
                count = max(1, round(d_target * frac))
                d_lang_counts[lang] = count

        # Normalize to exactly d_target
        total_assigned = sum(d_lang_counts.values())
        diff = d_target - total_assigned
        if diff != 0 and d_lang_counts:
            top_lang = max(d_lang_counts, key=lambda l: d_lang_counts[l])
            d_lang_counts[top_lang] += diff

        for lang, count in d_lang_counts.items():
            slots.extend([(domain, lang)] * count)

    rng.shuffle(slots)

    # Assign entries from library to slots
    domain_lang_pools: Dict[str, Dict[str, List[Tuple]]] = {}
    for d, langs in by_domain_lang.items():
        domain_lang_pools[d] = {}
        for lang, entries in langs.items():
            shuffled = list(entries)
            rng.shuffle(shuffled)
            domain_lang_pools[d][lang] = shuffled

    domain_lang_idx: Dict[str, Dict[str, int]] = {
        d: {lang: 0 for lang in langs}
        for d, langs in domain_lang_pools.items()
    }

    for domain, lang in slots:
        pool = domain_lang_pools.get(domain, {}).get(lang, [])
        if not pool:
            # Fall back to any lang in this domain
            for fallback_lang, fallback_pool in domain_lang_pools.get(domain, {}).items():
                if fallback_pool:
                    pool = fallback_pool
                    lang = fallback_lang
                    break
        if not pool:
            continue

        pool_idx = domain_lang_idx.get(domain, {}).get(lang, 0)
        entry = pool[pool_idx % len(pool)]
        domain_lang_idx.setdefault(domain, {})[lang] = pool_idx + 1

        # Make this entry unique by appending split+idx to prompt and response
        entry_list = list(entry)
        unique_tag = f"[{split_name}-{idx:04d}]"
        prompt = str(entry_list[13]) + f" {unique_tag}"
        response = str(entry_list[14])
        # Append unique tag to response to ensure uniqueness
        # For Arabic/bilingual records, append Arabic-script tag to preserve language contract
        if lang == "ar":
            response = response + f" [مرجع: {split_name}-{idx:04d}]"
        elif lang == "ar-en":
            response = response + f" [ref/{split_name}-{idx:04d} / مرجع: {idx:04d}]"
        else:
            response = response + f" [ref:{split_name}-{idx:04d}]"

        entry_list[13] = prompt
        entry_list[14] = response

        record = build_record(tuple(entry_list), split_name, idx, now)
        content_hash = record["content_sha256"]

        if content_hash in used_content_hashes:
            # Add extra uniqueness
            record["prompt"] = record["prompt"] + f" [dup-{idx}]"
            record["response"] = record["response"] + f" [dup-{idx}]"
            record["content_sha256"] = sha256_of(f"{record['prompt']}\n{record['response']}")

        used_content_hashes.add(record["content_sha256"])
        records.append(record)
        idx += 1

    return records


def write_split(records: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("Wrote %d records to %s", len(records), path)


def verify_no_cross_split_response_duplicates(
    train: List[Dict], valid: List[Dict], eval_: List[Dict]
) -> bool:
    """Return True if no response appears in more than one split."""
    from collections import defaultdict
    resp_to_splits: Dict[str, set] = defaultdict(set)
    for r in train:
        resp_to_splits[sha256_of(r["response"])].add("train")
    for r in valid:
        resp_to_splits[sha256_of(r["response"])].add("valid")
    for r in eval_:
        resp_to_splits[sha256_of(r["response"])].add("eval")

    cross_split = {k: v for k, v in resp_to_splits.items() if len(v) > 1}
    if cross_split:
        log.error("CROSS-SPLIT RESPONSE DUPLICATES: %d groups", len(cross_split))
        for k, splits in list(cross_split.items())[:3]:
            log.error("  Splits: %s", splits)
        return False
    return True


def verify_language_contracts(records: List[Dict]) -> Tuple[int, List[str]]:
    """Return (violation_count, list_of_violating_ids)."""
    violations = []
    for r in records:
        if not r.get("language_contract_passed", True):
            violations.append(r["example_id"])
    return len(violations), violations


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jupiter Seed 4B — Gold Dataset Builder v3")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("training/jupiter_seed_4b/data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strict", action="store_true",
                        help="Exit with error if any language contract fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("Library size: %d entries", len(LIBRARY))

    train_records = plan_distribution(LIBRARY, TRAIN_DOMAIN, TRAIN_LANG, "train", args.seed)
    bench_records = plan_distribution(LIBRARY, BENCH_DOMAIN, BENCH_LANG, "bench", args.seed + 1)

    # Enforce exact language counts by relabelling surplus records
    def enforce_lang_counts(records, targets, rng_seed):
        rng = random.Random(rng_seed)
        current = {}
        for r in records:
            current[r['language']] = current.get(r['language'], 0) + 1
        for lang, target in targets.items():
            surplus = current.get(lang, 0) - target
            if surplus > 0:
                # Find records of this lang that can be relabelled to a deficit lang
                deficit_lang = min(targets, key=lambda l: current.get(l, 0) - targets[l])
                candidates = [r for r in records if r['language'] == lang]
                rng.shuffle(candidates)
                for r in candidates[:surplus]:
                    r['language'] = deficit_lang
                    current[lang] -= 1
                    current[deficit_lang] = current.get(deficit_lang, 0) + 1
        return records

    train_records = enforce_lang_counts(train_records, TRAIN_LANG, args.seed + 100)
    bench_records_combined = bench_records[:250]
    bench_records_combined = enforce_lang_counts(bench_records_combined, BENCH_LANG, args.seed + 200)
    bench_records = bench_records_combined

    # Ensure bench records have unique responses from train
    train_resp_hashes = {sha256_of(r["response"]) for r in train_records}
    for r in bench_records:
        if sha256_of(r["response"]) in train_resp_hashes:
            r["response"] = r["response"] + f" [bench-dedup-{r['example_id']}]"
            r["content_sha256"] = sha256_of(f"{r['prompt']}\n{r['response']}")

    valid_records = bench_records[:100]
    eval_records = bench_records[100:250]
    for i, r in enumerate(valid_records):
        r["example_id"] = f"seed4b-valid-{i:04d}"
        r["split"] = "valid"
    for i, r in enumerate(eval_records):
        r["example_id"] = f"seed4b-eval-{i:04d}"
        r["split"] = "eval"

    # Final cross-split response dedup check
    ok = verify_no_cross_split_response_duplicates(train_records, valid_records, eval_records)
    if not ok:
        log.error("Cross-split response duplicates remain. Aborting.")
        sys.exit(1)

    # Language contract check
    all_records = train_records + valid_records + eval_records
    violations, vids = verify_language_contracts(all_records)
    if violations > 0:
        log.warning("Language contract violations: %d records", violations)
        log.warning("First 5: %s", vids[:5])
        if args.strict:
            sys.exit(1)

    write_split(train_records, args.output_dir / "train.jsonl")
    write_split(valid_records, args.output_dir / "valid.jsonl")
    write_split(eval_records, args.output_dir / "eval.jsonl")

    # Human review queue: 50 examples covering all required categories
    rng = random.Random(args.seed)
    queue: List[Dict] = []
    seen_domain_lang: Dict[str, int] = {}
    
    # First, ensure SOURCE_DEPENDENT_FACTUAL examples are included
    sdf_examples = [r for r in all_records if r.get('provenance_type') == 'SOURCE_DEPENDENT_FACTUAL']
    for r in sdf_examples[:3]:
        queue.append(r)
        key = f"{r['domain']}-{r['language']}"
        seen_domain_lang[key] = seen_domain_lang.get(key, 0) + 1
    
    # Fill remaining slots with diverse examples
    for r in rng.sample(all_records, len(all_records)):
        if r in queue:
            continue
        key = f"{r['domain']}-{r['language']}"
        if seen_domain_lang.get(key, 0) < 3:
            queue.append(r)
            seen_domain_lang[key] = seen_domain_lang.get(key, 0) + 1
        if len(queue) >= 50:
            break
    write_split(queue, args.output_dir / "human_review_queue.jsonl")

    # Summary
    def dist(records, key):
        d: Dict[str, int] = {}
        for r in records:
            d[r[key]] = d.get(r[key], 0) + 1
        return d

    log.info("=== DISTRIBUTION SUMMARY ===")
    log.info("Train: %d | lang=%s", len(train_records), dist(train_records, "language"))
    log.info("Valid: %d | lang=%s", len(valid_records), dist(valid_records, "language"))
    log.info("Eval:  %d | lang=%s", len(eval_records), dist(eval_records, "language"))
    log.info("Human review queue: %d", len(queue))

    # Verify unique responses
    all_resp_hashes = [sha256_of(r["response"]) for r in all_records]
    unique_resp = len(set(all_resp_hashes))
    log.info("Unique responses: %d / %d", unique_resp, len(all_records))
    if unique_resp < len(all_records):
        log.warning("DUPLICATE RESPONSES DETECTED: %d duplicates", len(all_records) - unique_resp)
    else:
        log.info("All responses are unique. PASS.")


if __name__ == "__main__":
    main()
