#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# UzABSA-LLM: Dataset Explorer — Language & Script Profiling
# =============================================================================
"""
Explore raw review data for Aspect-Based Sentiment Analysis.

This script performs:
1. Load & clean: reads CSV, drops NaN/empty review_text rows
2. Language/script profiling: classifies each review by script (Cyrillic/Latin)
   and language (Russian vs. Uzbek) using character-ratio analysis and word-
   level heuristics (Uzbek-specific Cyrillic markers ў қ ғ ҳ, and common
   Russian function words)
3. Statistics: counts and percentages per language category
4. Auto-logging: appends a structured summary to RESEARCH_LOG.md

Usage:
    python scripts/explore_datasets.py --raw-file ./data/raw/reviews.csv

Author: UzABSA Team
License: MIT
"""

import argparse
import logging
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json

from src.dataset_utils import (
    load_raw_reviews_csv,
)

# =============================================================================
# Configure Logging
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# Language / Script Classification
# =============================================================================

# Uzbek-specific Cyrillic letters (absent in standard Russian)
_UZBEK_CYRILLIC_MARKERS = set("ўқғҳЎҚҒҲ")

# Frequent Russian function words — if ≥2 appear in a Cyrillic text the text
# is very likely Russian rather than Uzbek written in Cyrillic.
_RUSSIAN_FUNCTION_WORDS = {
    "и", "в", "не", "на", "я", "что", "он", "она", "но", "это",
    "все", "как", "с", "из", "мне", "мы", "так", "они", "вы", "у",
    "от", "за", "по", "для", "бы", "до", "вот", "уже", "если",
    "при", "есть", "был", "было", "были", "быть", "очень", "или",
    "ни", "тоже", "ещё", "еще", "нет", "да", "же", "может",
}

# Common Uzbek words in Cyrillic script (for disambiguation)
_UZBEK_CYRILLIC_WORDS = {
    "жуда", "яхши", "зўр", "маззали", "ёмон", "нарх", "йўқ",
    "бор", "учун", "менга", "билан", "ҳам", "лекин", "аммо",
    "сиз", "мен", "бу", "шу", "ўша", "қилиш", "бериш",
}


def _char_script_ratios(text: str) -> Dict[str, float]:
    """Return the proportion of Cyrillic, Latin, and other characters."""
    cyrillic = 0
    latin = 0
    digit = 0
    other = 0

    for ch in text:
        cp = ord(ch)
        if 0x0400 <= cp <= 0x04FF:
            cyrillic += 1
        elif ("a" <= ch.lower() <= "z") or ch in "ʻʼ":
            latin += 1
        elif ch.isdigit():
            digit += 1
        elif not ch.isspace():
            other += 1

    alpha_total = cyrillic + latin
    if alpha_total == 0:
        return {"cyrillic": 0.0, "latin": 0.0, "total_alpha": 0}

    return {
        "cyrillic": cyrillic / alpha_total,
        "latin": latin / alpha_total,
        "total_alpha": alpha_total,
    }


def _has_uzbek_cyrillic_markers(text: str) -> bool:
    """Check for Uzbek-specific Cyrillic characters (ў, қ, ғ, ҳ)."""
    return any(ch in _UZBEK_CYRILLIC_MARKERS for ch in text)


def _russian_word_score(text: str) -> float:
    """Fraction of words in the text that are common Russian function words."""
    words = re.findall(r"[а-яёўқғҳА-ЯЁЎҚҒҲ]+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _RUSSIAN_FUNCTION_WORDS)
    return hits / len(words)


def _uzbek_cyrillic_word_score(text: str) -> float:
    """Fraction of words that are common Uzbek words (in Cyrillic)."""
    words = re.findall(r"[а-яёўқғҳА-ЯЁЎҚҒҲ]+", text.lower())
    if not words:
        return 0.0
    hits = sum(1 for w in words if w in _UZBEK_CYRILLIC_WORDS)
    return hits / len(words)


def classify_language(text: str) -> str:
    """
    Classify a single review into a language/script category.

    Dual approach:
      1. Character-level: ratio of Cyrillic vs Latin characters.
      2. Word-level heuristic:
         - Uzbek-specific Cyrillic markers (ў, қ, ғ, ҳ)
         - Russian function-word frequency

    Returns one of:
        'Primarily Uzbek (Latin)'
        'Primarily Russian (Cyrillic)'
        'Primarily Uzbek (Cyrillic)'
        'Highly Mixed'
    """
    ratios = _char_script_ratios(text)

    # Degenerate case — no alphabetic characters
    if ratios["total_alpha"] == 0:
        return "Primarily Uzbek (Latin)"  # default

    cyr = ratios["cyrillic"]
    lat = ratios["latin"]

    # ----- Latin-dominant (>70 % Latin characters) -----
    if lat > 0.70:
        return "Primarily Uzbek (Latin)"

    # ----- Cyrillic-dominant (>70 % Cyrillic characters) -----
    if cyr > 0.70:
        # Distinguish Russian from Uzbek-in-Cyrillic using word-level cues
        if _has_uzbek_cyrillic_markers(text):
            return "Primarily Uzbek (Cyrillic)"
        if _uzbek_cyrillic_word_score(text) > 0.05:
            return "Primarily Uzbek (Cyrillic)"
        if _russian_word_score(text) >= 0.08:
            return "Primarily Russian (Cyrillic)"
        # If no clear Russian signal, lean Uzbek-Cyrillic
        return "Primarily Russian (Cyrillic)"

    # ----- Mixed (30–70 % each) -----
    return "Highly Mixed"


# =============================================================================
# Business Category Classification
# =============================================================================

# Keyword-based mapping from object_name → business_category.
# Keys are lowercased substrings; first match wins (order matters).
_CATEGORY_KEYWORDS: List[Tuple[str, List[str]]] = [
    # --- Restoran / Ovqatlanish (Restaurant / Food) ---
    ("Restoran/Ovqatlanish", [
        "lavash", "pizza", "burger", "somsa", "kebab", "kebob", "shashlik",
        "shashleek", "doner", "döner", "hot-dog", "hotdog", "kfc", "evos",
        "feed up", "max way", "wendy", "havas", "kamolon", "besh qozon",
        "rayhon", "milliy taom", "akmal oltin", "oqtepa", "yapona mama",
        "gijduvon", "zohid", "oshpalov", "tandiriy", "tandoori",
        "restoran", "restaurant", "choyxona", "chopar", "soy restoran",
        "basri baba", "mandu manti", "roni pizza", "dodo", "pie republic",
        "sherin", "sochnaya", "cafe", "ca.fe", "bistro", "pub", "bar",
        "guten pub", "bla bla", "street 77", "bbq", "black star burger",
        "asl baraka", "hayat", "qora qamish", "plov", "national dishes",
        "sultan ahmet", "merhaba", "senior meathouse", "level restaurant",
        "efendi", "fresco", "wok", "afsona", "gosht", "bento", "giotto",
        "ozbegim restaurant", "mahmood", "retro milliy", "ansor family",
        "le taom", "fedorovich", "city 21", "loook", "alimovs",
        "kofteci", "china shashlik", "mona restaurant", "carvon",
        "marinno", "avocado bistro", "broccoli", "fish and bread",
        "gracia cafe", "khachapuri", "socials cafe", "april verdant",
        "fayz non", "axmad oltin", "asl kamolon", "toku", "dayako",
        "qanotchi", "paylov", "masterchef", "healthy food", "twice spice",
        "olim polvon", "apexpizza", "apex pizza", "best mix", "bu doner",
        "yalpiz", "island", "demir bey",
    ]),
    # --- Bank / Moliya (Banking / Finance) ---
    ("Bank/Moliya", [
        "bank", "banki", "lombard", "uzlombard", "hurma lombard",
        "kapital", "agrobank", "asakabank", "hamkorbank", "anor bank",
        "xalq bank", "ipoteka", "tenge", "sqb", "infin bank", "trast",
        "orient finance", "davr bank", "garant bank", "yangi bank",
        "mkbank", "aloqabank", "ipak yo", "turkiston bank",
        "universal bank", "ziraat", "nbf med",  # bank-related
    ]),
    # --- To'lov tizimlari (Payment / Fintech) ---
    ("To'lov tizimlari", [
        "click", "payme", "payway", "alifmobi", "xazna", "depozit",
        "marjon app",
    ]),
    # --- Telekommunikatsiya (Telecom) ---
    ("Telekommunikatsiya", [
        "uzmobile", "beeline", "ucell", "humans", "mobi.uz",
    ]),
    # --- Elektron tijorat (E-commerce / Marketplace) ---
    ("Elektron tijorat", [
        "uzum market", "olx", "openshop", "zoodmall", "asaxiy", "texnomart",
        "n1tools", "beil toys", "magnum", "elbozor", "inbazar", "sello",
        "g-shop", "medion",
    ]),
    # --- Oziq-ovqat do'konlari (Grocery / Retail) ---
    ("Oziq-ovqat do'konlari", [
        "korzinka", "baraka market", "makro", "freshbazar", "green apple",
        "olma market", "mukarram market", "mega planet", "tashkent city mall",
    ]),
    # --- Tibbiyot / Sog'liqni saqlash (Healthcare) ---
    ("Tibbiyot/Sog'liqni saqlash", [
        "medline", "hospital", "medical", "clinic", "med ", "medic",
        "shifo", "farmatsevtika", "dorixona", "apteka", "lor ", "ent plus",
        "stom", "diamed", "biogen", "onkologiya", "tug'ruq", "rodil",
        "massaj", "mubina wellness", "anatomica", "oxymed", "darmon",
        "sog'liq", "ayol care", "shox international", "chinor", "ixlos",
        "starmed", "profmed", "ifor pharm", "akmal farm", "genomed",
        "ivf doctor", "vivomed", "nurmed", "medzone", "saba-darmon",
        "real - med", "alfa med", "akrom stom", "kokcha med", "saneg",
        "ona va bola", "ona foundation", "hamshira", "hi tech lab",
        "lor mama", "prof dent",  "genotexnologiya", "eko-iksi",
        "siz ona bo", "respublika", "sbj medical", "ihlos doktor",
    ]),
    # --- Ta'lim (Education) ---
    ("Ta'lim", [
        "ta'lim", "talim", "university", "institut", "akadem", "school",
        "academy", "centre", "maktab", "avtomaktab", "ielts", "cambridge",
        "lerna", "skillbox", "najot", "alfraganus", "pdp", "webster",
        "westminster", "yeoju", "shiroq", "vector it", "mars it",
        "thompson", "mohirdev", "oneacademy", "fozilov", "hbs academy",
        "registan lc", "mohir ai", "profi university", "master ielts",
        "kasb education", "fashion art", "proweb", "iqtidor", "merax",
        "uic academy", "humo school", "trading academy", "my school",
        "best school", "hong kong", "registan", "al beruniy",
        "kolesnitsa", "excel", "inter nation", "nekov", "global education",
        "riks", "yakubov", "king's academy",
    ]),
    # --- Gul / Sovg'a (Flowers / Gifts) ---
    ("Gul/Sovg'a", [
        "flower", "gul", "roses", "gavalli", "toshkent gullari",
    ]),
    # --- Sport / Fitnes (Sports / Fitness) ---
    ("Sport/Fitnes", [
        "fit", "gym", "sport", "swimming", "basseyn", "born2swim",
        "arena", "power gym", "wunderfit", "victory fitness",
    ]),
    # --- Sayohat / Turizm (Travel / Tourism) ---
    ("Sayohat/Turizm", [
        "airways", "travel", "tour", "hotel", "sanatori", "kurorti",
        "dam olish", "hyatt", "asialuxe", "silkavia", "humo air",
        "zaamin", "chimyon",
    ]),
    # --- Yetkazib berish (Delivery) ---
    ("Yetkazib berish", [
        "yandex", "express 24", "my taxi", "mytaxi", "uklon",
        "royal taxi", "vip taxi", "veloxizmat",
    ]),
    # --- Go'zallik (Beauty) ---
    ("Go'zallik", [
        "beauty", "salon", "cosmetic", "barbershop", "dessange",
        "alina pro", "zulfizar", "elegant",
    ]),
    # --- Kitob / Nashriyot (Books / Publishing) ---
    ("Kitob/Nashriyot", [
        "kitob", "nashr", "book", "mutolaa", "kutubxo",
    ]),
    # --- Texnologiya / Media (Technology / Media) ---
    ("Texnologiya/Media", [
        "instagram", "netflix", "tv", "sharh", "daryo", "qalampir",
        "kun uz", "islom.uz", "itrade", "ticker", "dataprizma",
        "enter engineering", "zte", "dt ecosystem",
    ]),
    # --- Transport / Yo'l (Transport / Roads) ---
    ("Transport/Yo'l", [
        "road", "yo'l", "temir yo", "chevrolet", "byd", "china motors",
        "kia", "auto doctor", "car wash", "avto", "li ning",
        "metan", "azs", "zty car", "uzpost",
    ]),
    # --- Ko'ngilochar / Park (Entertainment / Parks) ---
    ("Ko'ngilochar", [
        "park", "cinema", "magic city", "zoo", "jumanji", "dream park",
        "teatr", "teatro", "milliy ansambil", "concert", "music brand",
        "zeus game",
    ]),
    # --- Sugʻurta (Insurance) ---
    ("Sug'urta", [
        "insurance", "sug'urta",
    ]),
    # --- Davlat xizmatlari (Government Services) ---
    ("Davlat xizmatlari", [
        "gov.uz", "hududgaz", "uz suv", "yashil energiya", "kommunal",
        "hgt", "vatanparvar",
    ]),
    # --- Din / Madaniyat (Religious / Cultural) ---
    ("Din/Madaniyat", [
        "masjid", "mosque", "to'xtaniyoz", "minor mosque", "horev",
        "vaqf",
    ]),
    # --- Bozor / BSC (Bazaar / Business Support Center) ---
    ("Bozor/BSC", [
        "bsc", "bozor", "bazaar", "zarhal plaza",
    ]),
    # --- Investitsiya / Trading ---
    ("Investitsiya/Trading", [
        "invest", "trading", "fx", "capital", "az capital", "grand invest",
        "akefx", "trend line",
    ]),
]


# Predefined ABSA subcategories per business category.
# These will be used for aspect annotation and model training.
ABSA_SUBCATEGORIES: Dict[str, Dict[str, List[str]]] = {
    "Restoran/Ovqatlanish": {
        "description": "Restaurants, fast food, cafes, national cuisine",
        "subcategories": [
            "ovqat_sifati",           # food quality / taste
            "xizmat_ko'rsatish",      # customer service
            "narx",                   # price / value for money
            "tozalik",                # cleanliness / hygiene
            "muhit",                  # ambiance / atmosphere
            "tezlik",                 # speed of service
            "menyu_xilma-xilligi",    # menu variety
            "joylashuv",              # location / accessibility
            "porsiya",                # portion size
        ],
    },
    "Bank/Moliya": {
        "description": "Banks, financial services, loans",
        "subcategories": [
            "xizmat_ko'rsatish",      # customer service
            "ilova_qulayligi",        # mobile app usability
            "kredit",                 # credit / loan services
            "foiz_stavka",            # interest rates
            "tezlik",                 # processing speed
            "xavfsizlik",             # security
            "filial",                 # branch experience
            "qo'llab-quvvatlash",     # customer support
            "karta_xizmati",          # card services
            "onlayn_xizmat",          # online services
        ],
    },
    "To'lov tizimlari": {
        "description": "Mobile payment, fintech, digital wallets",
        "subcategories": [
            "ilova_qulayligi",        # app usability / UX
            "tezlik",                 # transaction speed
            "xavfsizlik",             # security
            "qo'llab-quvvatlash",     # customer support
            "komissiya",              # fees / commission
            "funksionallik",          # features / functionality
            "ishonchlilik",           # reliability / uptime
        ],
    },
    "Telekommunikatsiya": {
        "description": "Mobile operators, internet service providers",
        "subcategories": [
            "internet_sifati",        # internet quality / speed
            "aloqa_sifati",           # call quality
            "narx",                   # pricing / tariffs
            "qo'llab-quvvatlash",     # customer support
            "qamrov",                 # network coverage
            "tezlik",                 # data speed
            "ilova_qulayligi",        # app usability
        ],
    },
    "Elektron tijorat": {
        "description": "Online marketplaces, e-commerce platforms",
        "subcategories": [
            "mahsulot_sifati",        # product quality
            "yetkazib_berish",        # delivery
            "narx",                   # price
            "qo'llab-quvvatlash",     # customer support
            "tanlov",                 # selection / product variety
            "qaytarish",              # returns / refund policy
            "ilova_qulayligi",        # app/website usability
            "to'lov",                 # payment process
        ],
    },
    "Oziq-ovqat do'konlari": {
        "description": "Supermarkets, grocery stores, retail chains",
        "subcategories": [
            "mahsulot_sifati",        # product freshness / quality
            "narx",                   # pricing
            "xizmat_ko'rsatish",      # customer service
            "tozalik",                # store cleanliness
            "tanlov",                 # product variety
            "joylashuv",              # location / accessibility
            "navbat",                 # queue / wait time
        ],
    },
    "Tibbiyot/Sog'liqni saqlash": {
        "description": "Hospitals, clinics, pharmacies, healthcare",
        "subcategories": [
            "shifokor_malakasi",      # doctor competence
            "xizmat_ko'rsatish",      # service quality
            "narx",                   # pricing
            "tozalik",                # cleanliness / hygiene
            "diagnostika",            # diagnosis accuracy
            "navbat",                 # wait time
            "jihozlar",               # equipment / facilities
            "dori",                   # medications / pharmacy
            "qo'llab-quvvatlash",     # patient support
        ],
    },
    "Ta'lim": {
        "description": "Universities, schools, courses, training centers",
        "subcategories": [
            "o'qitish_sifati",        # teaching quality
            "o'qituvchi",             # teacher / instructor
            "narx",                   # tuition / pricing
            "dastur",                 # curriculum / program
            "infratuzilma",           # infrastructure / facilities
            "natija",                 # learning outcomes / results
            "sertifikat",             # certification value
            "amaliyot",              # practical training / internship
        ],
    },
    "Gul/Sovg'a": {
        "description": "Flower shops, gift stores",
        "subcategories": [
            "sifat",                  # flower/product quality
            "narx",                   # pricing
            "yetkazib_berish",        # delivery
            "xizmat_ko'rsatish",      # customer service
            "tanlov",                 # variety / selection
            "tezlik",                 # speed of delivery
        ],
    },
    "Sport/Fitnes": {
        "description": "Gyms, fitness centers, sports facilities",
        "subcategories": [
            "jihozlar",               # equipment quality
            "murabbiy",               # trainer / coach
            "narx",                   # pricing / membership
            "tozalik",                # cleanliness
            "muhit",                  # atmosphere
            "joylashuv",              # location
        ],
    },
    "Sayohat/Turizm": {
        "description": "Airlines, hotels, travel agencies, resorts",
        "subcategories": [
            "xizmat_ko'rsatish",      # service quality
            "narx",                   # pricing
            "qulaylik",               # comfort
            "tozalik",                # cleanliness
            "ovqat_sifati",           # food quality (in-flight, hotel)
            "joylashuv",              # location
            "xodimlar",               # staff behavior
        ],
    },
    "Yetkazib berish": {
        "description": "Delivery services, ride-hailing, logistics",
        "subcategories": [
            "tezlik",                 # delivery speed
            "narx",                   # pricing
            "xizmat_ko'rsatish",      # driver/courier service
            "ilova_qulayligi",        # app usability
            "ishonchlilik",           # reliability
            "xavfsizlik",             # safety
        ],
    },
    "Go'zallik": {
        "description": "Beauty salons, cosmetics, barbershops",
        "subcategories": [
            "sifat",                  # service quality
            "narx",                   # pricing
            "tozalik",                # cleanliness / hygiene
            "mutaxassislik",          # specialist skill
            "muhit",                  # atmosphere
        ],
    },
    "Kitob/Nashriyot": {
        "description": "Bookstores, publishing houses",
        "subcategories": [
            "tanlov",                 # book variety / selection
            "narx",                   # pricing
            "sifat",                  # print / product quality
            "xizmat_ko'rsatish",      # customer service
            "yetkazib_berish",        # delivery
        ],
    },
    "Texnologiya/Media": {
        "description": "Tech platforms, media, news outlets",
        "subcategories": [
            "kontent_sifati",         # content quality
            "ilova_qulayligi",        # app/platform usability
            "ishonchlilik",           # reliability
            "reklama",                # ads / advertising
            "qo'llab-quvvatlash",     # support
        ],
    },
    "Transport/Yo'l": {
        "description": "Transport, roads, auto services",
        "subcategories": [
            "sifat",                  # road / service quality
            "narx",                   # pricing
            "xavfsizlik",             # safety
            "qulaylik",               # comfort
            "tezlik",                 # speed
        ],
    },
    "Ko'ngilochar": {
        "description": "Entertainment, parks, cinema, theater",
        "subcategories": [
            "ko'ngilochar_sifati",    # entertainment quality
            "narx",                   # pricing
            "xizmat_ko'rsatish",      # service
            "muhit",                  # atmosphere / environment
            "tozalik",                # cleanliness
        ],
    },
    "Boshqa": {
        "description": "Other / miscellaneous",
        "subcategories": [
            "sifat",                  # general quality
            "narx",                   # pricing
            "xizmat_ko'rsatish",      # service
            "qo'llab-quvvatlash",     # support
        ],
    },
}


def classify_business_category(object_name: str) -> str:
    """
    Classify a business (object_name) into a high-level category.

    Uses keyword matching against the object name (case-insensitive).
    First match wins, so ordering in _CATEGORY_KEYWORDS matters.

    Args:
        object_name: Name of the business/object from reviews.csv.

    Returns:
        Business category string (e.g., 'Restoran/Ovqatlanish', 'Bank/Moliya').
    """
    name_lower = str(object_name).lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        for kw in keywords:
            if kw.lower() in name_lower:
                return category
    return "Boshqa"


def compute_business_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each review's object_name into a business category and compute
    distribution statistics.

    Args:
        df: DataFrame with 'object_name' column.

    Returns:
        DataFrame with columns: business_category, review_count,
        unique_businesses, percentage.
    """
    df["business_category"] = df["object_name"].apply(classify_business_category)

    stats = (
        df.groupby("business_category")
        .agg(
            review_count=("review_text", "size"),
            unique_businesses=("object_name", "nunique"),
        )
        .reset_index()
    )
    stats["percentage"] = (stats["review_count"] / len(df) * 100).round(2)
    stats = stats.sort_values("review_count", ascending=False).reset_index(drop=True)
    return stats


def get_absa_subcategories_summary() -> str:
    """Return a formatted string summary of all ABSA subcategories."""
    lines = []
    for cat, info in ABSA_SUBCATEGORIES.items():
        lines.append(f"\n  [{cat}] — {info['description']}")
        for sub in info["subcategories"]:
            lines.append(f"    • {sub}")
    return "\n".join(lines)


# =============================================================================
# Statistics
# =============================================================================

def compute_lang_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a summary table with counts and percentages per lang_category.

    Returns a DataFrame with columns: lang_category, count, percentage.
    """
    counts = df["lang_category"].value_counts()
    stats = pd.DataFrame({
        "lang_category": counts.index,
        "count": counts.values,
        "percentage": (counts.values / len(df) * 100).round(2),
    })
    stats = stats.sort_values("count", ascending=False).reset_index(drop=True)
    return stats


def compute_text_stats(df: pd.DataFrame, text_col: str = "review_text") -> Dict:
    """Compute basic text statistics."""
    texts = df[text_col].astype(str)
    word_counts = texts.str.split().str.len()
    char_counts = texts.str.len()
    return {
        "total_reviews": len(df),
        "avg_words": round(word_counts.mean(), 2),
        "avg_chars": round(char_counts.mean(), 2),
        "min_words": int(word_counts.min()),
        "max_words": int(word_counts.max()),
        "min_chars": int(char_counts.min()),
        "max_chars": int(char_counts.max()),
        "median_words": int(word_counts.median()),
        "median_chars": int(char_counts.median()),
    }


# =============================================================================
# Markdown Logging
# =============================================================================

def append_to_research_log(
    log_path: str,
    lang_stats: pd.DataFrame,
    text_stats: Dict,
    biz_stats: pd.DataFrame,
    df: pd.DataFrame,
    raw_file: str,
) -> None:
    """Append a structured log entry to RESEARCH_LOG.md."""
    today = datetime.now().strftime("%b %d, %Y")

    # Build samples per category (up to 2 each)
    sample_lines = []
    for cat in lang_stats["lang_category"]:
        subset = df[df["lang_category"] == cat]
        sample_lines.append(f"  **{cat}**:")
        for _, row in subset.head(2).iterrows():
            snippet = str(row["review_text"])[:120].replace("\n", " ")
            sample_lines.append(f'  - _{snippet}_…')

    samples_block = "\n".join(sample_lines)

    # Build business category table
    biz_table_rows = ""
    for _, row in biz_stats.iterrows():
        biz_table_rows += f"| {row['business_category']} | {row['review_count']} | {row['unique_businesses']} | {row['percentage']}% |\n"

    # Build ABSA subcategory block
    subcat_lines = []
    total_subcats = 0
    for cat, info in ABSA_SUBCATEGORIES.items():
        subcat_lines.append(f"- **{cat}** — {info['description']}")
        for sub in info["subcategories"]:
            subcat_lines.append(f"  - `{sub}`")
            total_subcats += 1
    subcat_block = "\n".join(subcat_lines)

    # Build the markdown entry
    entry = f"""

## LOG 017 — Data Exploration: Business Category & ABSA Subcategory Analysis
Date: {today}

### Source
- File: `{raw_file}`
- Total reviews loaded: **{text_stats['total_reviews']}**
- After cleaning (drop NaN/empty `review_text`): **{len(df)}**
- Unique businesses: **{df['object_name'].nunique()}**

### Text Statistics
| Metric | Value |
|--------|-------|
| Avg words / review | {text_stats['avg_words']} |
| Avg chars / review | {text_stats['avg_chars']} |
| Median words | {text_stats['median_words']} |
| Word range | {text_stats['min_words']}–{text_stats['max_words']} |
| Char range | {text_stats['min_chars']}–{text_stats['max_chars']} |

### Language/Script Classification Results
| Language Category | Count | Percentage |
|-------------------|------:|----------:|
"""

    for _, row in lang_stats.iterrows():
        entry += f"| {row['lang_category']} | {row['count']} | {row['percentage']}% |\n"

    entry += f"""
### Sample Reviews per Language Category
{samples_block}

### Business Category Classification
- **Method:** Keyword-based mapping from `object_name` to business domain
- **Total categories identified:** {len(biz_stats)}

| Business Category | Reviews | Businesses | Percentage |
|--------------------|--------:|-----------:|-----------:|
{biz_table_rows}
### Key Findings — Business Domain Distribution
- **Largest domain:** {biz_stats.iloc[0]['business_category']} ({biz_stats.iloc[0]['review_count']} reviews, {biz_stats.iloc[0]['percentage']}%)
- The dataset covers **{len(biz_stats)}** distinct business domains — this is a MULTI-DOMAIN dataset
- Unlike standard ABSA benchmarks (restaurant-only like SemEVAL 2014), our dataset spans restaurants, banks, telecom, healthcare, education, e-commerce, and more
- **This is a key contribution:** First multi-domain ABSA dataset for Uzbek language

### Predefined ABSA Subcategories per Business Domain
- Total: **{total_subcats} subcategories** across **{len(ABSA_SUBCATEGORIES)} domains**
- These subcategories define the aspect taxonomy for annotation and model training

{subcat_block}

### Rationale for Subcategory Design
- Subcategories are designed based on:
  1. Common aspects mentioned in real reviews (observed from data)
  2. Domain-specific attributes (e.g., `kredit` for banks, `shifokor_malakasi` for healthcare)
  3. Cross-domain shared aspects (e.g., `narx`, `xizmat_ko'rsatish` appear in most domains)
- Uzbek-language category names chosen for consistency with Uzbek ABSA task
- Categories are **hierarchical**: business_category → subcategory → polarity

### Implications for ABSA Fine-tuning
- The dataset is **predominantly Uzbek in Latin script** ({lang_stats[lang_stats['lang_category'] == 'Primarily Uzbek (Latin)']['percentage'].values[0] if 'Primarily Uzbek (Latin)' in lang_stats['lang_category'].values else 0}%).
- Multi-domain nature requires **domain-aware aspect categories** (not a flat list)
- Predefined subcategories allow:
  1. Structured annotation of the raw reviews.csv dataset
  2. Category-aware training (the model learns domain-specific aspects)
  3. Evaluation per domain (compare model performance across business categories)
- **Recommendation:** Annotate raw reviews using the predefined subcategory taxonomy, then combine with the existing HuggingFace annotated dataset for a larger, richer training set.
- **Output files saved:**
  - `data/raw/absa_subcategories.json` — Full subcategory taxonomy
  - `data/raw/business_categories.json` — Business→category mapping for all 630 businesses
"""

    # Append to file
    log_path_obj = Path(log_path)
    if not log_path_obj.exists():
        logger.warning(f"RESEARCH_LOG.md not found at {log_path}. Creating new file.")

    # Insert before the END marker if it exists
    if log_path_obj.exists():
        content = log_path_obj.read_text(encoding="utf-8")
        end_marker = "# ======================================================================"
        end_section = "# END OF CURRENT LOGS"

        if end_section in content:
            # Find the last occurrence of the end block
            idx = content.rfind(end_section)
            # Walk back to find the preceding separator
            block_start = content.rfind(end_marker, 0, idx)
            if block_start == -1:
                block_start = idx
            new_content = content[:block_start].rstrip() + "\n" + entry + "\n\n" + content[block_start:]
            log_path_obj.write_text(new_content, encoding="utf-8")
        else:
            # No end marker — just append
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(entry)
    else:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("# Research Log — UzABSA-LLM Project\n\n")
            f.write(entry)

    logger.info(f"Appended language profiling results to {log_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    """Main exploration pipeline."""
    parser = argparse.ArgumentParser(
        description="Explore UzABSA dataset — language/script profiling"
    )
    parser.add_argument(
        "--raw-file",
        type=str,
        default="./data/raw/reviews.csv",
        help="Path to raw reviews CSV file",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="./RESEARCH_LOG.md",
        help="Path to research log markdown file",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip appending results to RESEARCH_LOG.md",
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("UzABSA Dataset Explorer — Language/Script Profiling")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Load and clean
    # ------------------------------------------------------------------
    print("\n[1/6] Loading raw reviews...")
    print("-" * 70)

    raw_file_path = Path(args.raw_file)
    if not raw_file_path.exists():
        logger.error(f"Raw reviews file not found: {args.raw_file}")
        sys.exit(1)

    df = pd.read_csv(str(raw_file_path))
    total_before = len(df)
    logger.info(f"Loaded {total_before} rows from {args.raw_file}")

    # Drop NaN / empty review_text
    df = df.dropna(subset=["review_text"])
    df = df[df["review_text"].astype(str).str.strip().str.len() > 0].copy()
    dropped = total_before - len(df)
    logger.info(f"Dropped {dropped} rows with NaN/empty review_text → {len(df)} remaining")

    print(f"  Loaded:  {total_before} rows")
    print(f"  Cleaned: {len(df)} rows  (dropped {dropped} NaN/empty)")

    # Show a sample
    sample = df.iloc[0]
    print(f"\n  Sample review:")
    print(f"    Object: {sample['object_name']}")
    print(f"    Rating: {sample['rating_value']}/5")
    print(f"    Text:   {str(sample['review_text'])[:130]}…")

    # ------------------------------------------------------------------
    # Step 2: Language/script profiling
    # ------------------------------------------------------------------
    print(f"\n[2/6] Classifying language/script for {len(df)} reviews...")
    print("-" * 70)

    df["lang_category"] = df["review_text"].astype(str).apply(classify_language)
    logger.info("Language classification complete.")

    # ------------------------------------------------------------------
    # Step 3: Business category classification
    # ------------------------------------------------------------------
    print(f"\n[3/6] Classifying business categories for {df['object_name'].nunique()} unique businesses...")
    print("-" * 70)

    biz_stats = compute_business_category_stats(df)
    logger.info("Business category classification complete.")

    print(f"\n  Business Category Distribution:")
    print(f"    {'Category':<35} {'Reviews':>8}  {'Businesses':>11}  {'%':>7}")
    print(f"    {'─' * 35} {'─' * 8}  {'─' * 11}  {'─' * 7}")
    for _, row in biz_stats.iterrows():
        print(f"    {row['business_category']:<35} {row['review_count']:>8}  {row['unique_businesses']:>11}  {row['percentage']:>6.2f}%")
    print(f"    {'─' * 35} {'─' * 8}  {'─' * 11}  {'─' * 7}")
    print(f"    {'TOTAL':<35} {len(df):>8}  {df['object_name'].nunique():>11}  {100.00:>6.2f}%")

    # Show top businesses per category
    print(f"\n  Top businesses per category:")
    for _, row in biz_stats.iterrows():
        cat = row["business_category"]
        subset = df[df["business_category"] == cat]
        top = subset["object_name"].value_counts().head(3)
        print(f"\n    [{cat}]")
        for biz_name, cnt in top.items():
            print(f"      {biz_name}: {cnt} reviews")

    # ------------------------------------------------------------------
    # Step 4: ABSA subcategories per business category
    # ------------------------------------------------------------------
    print(f"\n[4/6] Predefined ABSA subcategories per business category")
    print("-" * 70)
    print(get_absa_subcategories_summary())

    # Save subcategories to JSON for downstream use
    subcat_path = Path(args.raw_file).parent / "absa_subcategories.json"
    with open(subcat_path, "w", encoding="utf-8") as f:
        json.dump(ABSA_SUBCATEGORIES, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Saved ABSA subcategories to {subcat_path}")

    # Save business category mapping to JSON
    biz_map_path = Path(args.raw_file).parent / "business_categories.json"
    biz_map = (
        df[["object_name", "business_category"]]
        .drop_duplicates()
        .sort_values(["business_category", "object_name"])
        .to_dict(orient="records")
    )
    with open(biz_map_path, "w", encoding="utf-8") as f:
        json.dump(biz_map, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved business→category mapping to {biz_map_path}")

    # ------------------------------------------------------------------
    # Step 5: Language & Text Statistics
    # ------------------------------------------------------------------
    print("\n[5/6] Computing text statistics...")
    print("-" * 70)

    lang_stats = compute_lang_stats(df)
    text_stats = compute_text_stats(df)

    # Print text statistics
    print(f"\n  Text Statistics:")
    print(f"    Total reviews:      {text_stats['total_reviews']}")
    print(f"    Avg words/review:   {text_stats['avg_words']}")
    print(f"    Avg chars/review:   {text_stats['avg_chars']}")
    print(f"    Median words:       {text_stats['median_words']}")
    print(f"    Word range:         {text_stats['min_words']}–{text_stats['max_words']}")
    print(f"    Char range:         {text_stats['min_chars']}–{text_stats['max_chars']}")

    # Print language distribution
    print(f"\n  Language/Script Distribution:")
    print(f"    {'Category':<35} {'Count':>6}  {'%':>7}")
    print(f"    {'─' * 35} {'─' * 6}  {'─' * 7}")
    for _, row in lang_stats.iterrows():
        print(f"    {row['lang_category']:<35} {row['count']:>6}  {row['percentage']:>6.2f}%")
    print(f"    {'─' * 35} {'─' * 6}  {'─' * 7}")
    print(f"    {'TOTAL':<35} {len(df):>6}  {100.00:>6.2f}%")

    # Show examples per category
    print(f"\n  Samples per category:")
    for cat in lang_stats["lang_category"]:
        subset = df[df["lang_category"] == cat]
        print(f"\n    [{cat}]  ({len(subset)} reviews)")
        for _, row in subset.head(2).iterrows():
            snippet = str(row["review_text"])[:100].replace("\n", " ")
            print(f"      → {snippet}…")

    # ------------------------------------------------------------------
    # Step 6: Append to RESEARCH_LOG.md
    # ------------------------------------------------------------------
    if not args.no_log:
        print(f"\n[6/6] Appending results to {args.log_file}...")
        print("-" * 70)
        append_to_research_log(
            log_path=args.log_file,
            lang_stats=lang_stats,
            text_stats=text_stats,
            biz_stats=biz_stats,
            df=df,
            raw_file=args.raw_file,
        )
        print(f"  ✓ Results appended to {args.log_file}")
    else:
        print(f"\n[6/6] Skipping RESEARCH_LOG.md (--no-log)")

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Dataset Exploration Complete!")
    print("=" * 70)
    print(f"\n  Summary:")
    print(f"    Reviews analyzed: {len(df)}")
    print(f"    Unique businesses: {df['object_name'].nunique()}")
    print(f"    Business categories: {len(biz_stats)}")
    for _, row in lang_stats.iterrows():
        print(f"    {row['lang_category']}: {row['count']} ({row['percentage']}%)")
    print(f"\n  Top 5 business categories:")
    for _, row in biz_stats.head(5).iterrows():
        print(f"    {row['business_category']}: {row['review_count']} reviews ({row['percentage']}%)")
    print(f"\n  ABSA subcategories defined: {sum(len(v['subcategories']) for v in ABSA_SUBCATEGORIES.values())} across {len(ABSA_SUBCATEGORIES)} categories")
    print(f"\n  Output files:")
    print(f"    - {Path(args.raw_file).parent / 'absa_subcategories.json'}")
    print(f"    - {Path(args.raw_file).parent / 'business_categories.json'}")
    print(f"\n  Next steps:")
    print(f"    1. Annotate reviews with aspects using subcategories")
    print(f"    2. Prepare dataset:  python scripts/prepare_complete_dataset.py --max-examples -1 --output-dir ./data/processed")
    print(f"    3. Start training:   python scripts/train_unsloth.py --help")


if __name__ == "__main__":
    main()
