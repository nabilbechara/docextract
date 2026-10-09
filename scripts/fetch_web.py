"""Download a curated set of real bill/receipt/invoice photos from Wikimedia Commons (free licenses).

Ground truth for these was labelled by hand by reading each image (data/real/web/*.json).
Usage: python scripts/fetch_web.py data/real/web
"""
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

FILES = {
    "nepal_restaurant": "2015-03-09 Dalbath Receipt of Yin Yang Restaurant in Thamel, Kathmandu, Nepal ダルバート DSCF6295.JPG",
    "fine_meal_bill": "A fine bill for a fine meal!.jpg",
    "vienna_water": "2007.08.15.Kassazettel.WienerWasser.JPG",
    "abbott_holder_1969": "Abbott and Holder receipt - 21 August 1969.jpg",
    "dutch_invoice_hengst": "Factuur van De Hengst AAIGEM-030.jpg",
    "french_facture_bergeron": "Facture etablissements Bergeron.jpg",
    "bulgarian_invoice": "Ахмед Джевдет и Нафиз фактура.jpg",
    "mysore_restaurant": "Bill in a local restaurant, Mysore, Karnataka.jpg",
    "cash_carry_memo": "Memo Cash and Carry.jpg",
    "pikes_peak_1889": "Invoice for labor and supplies submitted to Pikes Peak Toll Road, 1889 (NBY 18088).jpg",
    "macclesfield_inn": "Macclesfield Receipt Bull and Gate Inn.jpg",
    "petra_basin": "Basin Restaurant Petra check 2021.jpg",
    "bill_vishwanand": "Bill Vishwanand 2026-01-28.jpg",
    "pizzeria_miana": "Pizzeria Miana smaku check 2021.jpg",
    "castorama_wroclaw": "Castorama Wroclaw Krzywoustego receipt 2021.jpg",
    "family_dollar": "Family Dollar store receipt.jpg",
    "lawson_indonesia": "Receipt from Lawson in South Tangerang, Indonesia with QRIS CPM (2025-09-02).jpg",
    "atb_chernihiv": "Чек АТБ Чернігів.JPG",
    "malmo_1899": "Malmo & Co receipt, 1899 (MOHAI 11884).jpg",
    "sir_alec_rose": "Receipt from Sir Alec Rose, Fruiterer - 53632044650.jpg",
}
UA = {"User-Agent": "docextract-eval/0.1 (personal testing; low volume)"}

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/real/web")
out.mkdir(parents=True, exist_ok=True)
for key, title in FILES.items():
    dst = out / f"{key}.jpg"
    if dst.exists():
        continue
    url = "https://commons.wikimedia.org/wiki/Special:FilePath/" + urllib.parse.quote(title) + "?width=1600"
    for i in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                dst.write_bytes(r.read())
            print(f"{key}: {dst.stat().st_size // 1024} KB", flush=True)
            break
        except Exception as e:  # noqa: BLE001
            print(f"{key}: retry {i + 1} {e}", flush=True)
            time.sleep(5 * (i + 1))
