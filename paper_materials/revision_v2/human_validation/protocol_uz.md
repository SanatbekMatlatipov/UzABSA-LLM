# Inson tomonidan baholash bo'yicha qo'llanma (UzABSA-LLM)

**Maqsad.** Model tomonidan avtomatik ajratilgan aspekt-hissiyot juftliklarining sifatini
mutaxassis (ona tili) baholovchilar orqali tekshirish. Sizning baholaringiz maqolada
(a) baholovchilar o'rtasidagi kelishuvni, (b) sun'iy intellekt "hakami" (GPT-4o-mini)
baholarining ishonchliligini, va (c) chiqarilgan ma'lumotlar to'plamining haqiqiy sifatini
o'lchash uchun ishlatiladi.

Har bir baholovchi ishni **mustaqil** bajaradi (bir-biringiz bilan maslahatlashmang).

---

## Asosiy tushunchalar

- **Aspekt termini (term):** matnda fikr bildirilgan aniq narsa/xususiyat.
  Masalan: *ovqat, narx, xizmat, ilova, yetkazib berish, shifokor*.
- **Hissiyot (polarity):** shu aspektga bildirilgan munosabat. To'rt qiymatdan biri:
  - `positive` — ijobiy
  - `negative` — salbiy
  - `neutral` — betaraf (fikr bor, lekin ijobiy/salbiy emas)
  - `conflict` — bir aspekt haqida ham ijobiy, ham salbiy fikr bor

---

## 1-topshiriq: Model bashoratlarini baholash (`rubric_template.csv`)

Bu faylda har bir qatorda **sharh matni** va **model ajratgan aspektlar** ko'rsatilgan.
Model natijasini quyidagi **beshta o'lcham** bo'yicha **1 dan 5 gacha** baholang
(1 = juda yomon, 5 = a'lo). Baholaringizni tegishli ustunlarga yozing.

| Ustun | O'lcham | Nimani baholaysiz |
|---|---|---|
| `completeness_1_5` | To'liqlik | Matndagi barcha muhim fikrlar qamrab olinganmi? |
| `accuracy_1_5` | Aniqlik | Ajratilgan terminlar haqiqatan matnda bormi (model o'ylab topmaganmi)? |
| `sentiment_1_5` | Hissiyot to'g'riligi | Polarity (ijobiy/salbiy/...) to'g'ri belgilanganmi? |
| `relevance_1_5` | Muvofiqlik | Kategoriya va aspekt shu sharh kontekstiga mosmi? |
| `overall_1_5` | Umumiy | Umumiy sifat bahoyingiz |

- **Barcha ~150 qatorni** to'ldiring.
- Agar model **hech qanday aspekt topmagan** bo'lsa ("model predicted no aspects"), matnda
  haqiqatan aspekt bo'lsa `completeness`/`overall` past baho bering; matnda ham aspekt
  yo'q bo'lsa yuqori baho bering.
- Izohlaringizni (ixtiyoriy) `notes` ustuniga yozishingiz mumkin.

---

## 2-topshiriq: To'g'ri javoblarni yozish (`gold_template.csv`)

Bu **~80 ta** sharh uchun modelga qaramasdan, **o'zingiz** to'g'ri aspekt-hissiyot
juftliklarini yozing. Faqat `gold_aspects` ustunini to'ldiring.

**Format (juda muhim — aniq quyidagicha yozing):**

```
term :: polarity ;; term :: polarity ;; ...
```

- Har bir juftlik: `term :: polarity`
- Juftliklar orasida: ` ;; ` (bo'sh joy, ikki nuqta-vergul, bo'sh joy)
- Term — matndagi so'z(lar), polarity — `positive` / `negative` / `neutral` / `conflict`

**Misollar** (faylda `EXAMPLE_1`, `EXAMPLE_2` qatorlari — topshirishdan oldin ularni o'chiring):

| Matn | To'g'ri javob (`gold_aspects`) |
|---|---|
| "Ovqatlari mazali edi lekin narxi qimmat." | `ovqat :: positive ;; narx :: negative` |
| "Ilova sekin ishlaydi, xizmat yaxshi." | `ilova :: negative ;; xizmat :: positive` |

- Agar sharhda **umuman aspekt bo'lmasa**, `gold_aspects` ni **bo'sh** qoldiring.
- Iloji boricha matndagi **asl so'z shaklini** ishlating (masalan, "narxi" emas "narx" deб
  soddalashtirsangiz ham bo'ladi, lekin izchil bo'ling).

---

## Topshirish

To'ldirilgan ikkala faylni `paper_materials/revision_v2/human_validation/returned/`
papkasiga quyidagi nom bilan saqlang:

- `rubric_<ismingiz>.csv`  (masalan `rubric_annotator1.csv`)
- `gold_<ismingiz>.csv`

CSV faylni **UTF-8** kodlashda saqlang (Excel'da: *Save As → CSV UTF-8*). Ismlar
maqolada oshkor qilinmaydi (annotator1 / annotator2 deb yuritiladi).

Rahmat! Sizning ishingiz past-resursli o'zbek tili uchun ilmiy hissa qo'shadi.
