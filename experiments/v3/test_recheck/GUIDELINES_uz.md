# Test to'plamini qayta tekshirish: annotatorlar uchun qo'llanma

**Maqsad.** UzABSA test to'plamidagi (609 ta restoran sharhi) oltin (gold) belgilardagi xatolarni
topish va tuzatish. Har bir sharh uchun bir nechta **nomzod so'z** (`candidate_term`) berilgan. Siz har
biri shu sharhda aspekt termini ekanligini va uning hissiyotini (polarity) belgilaysiz.

**Mustaqil ishlang.** Boshqa annotator bilan maslahatlashmang, model natijalariga va maqolaga qaramang.
Nomzodlarning bir qismi eski oltin belgilardan, bir qismi modellardan olingan. Qaysi biri qayerdan
kelganini bilmaysiz, bu ataylab shunday qilingan. Har bir nomzodni faqat shu qo'llanma bo'yicha baholang.

**Vaqt.** Taxminan 3–4 soat (263 ta sharh, 575 ta nomzod). 2–3 marta bo'lib bajarsangiz bo'ladi.

---

## 1. Aspekt termini nima?

Aspekt termini — sharhda **so'z bilan nomlangan** va muallif u haqida gapirayotgan narsa:

- taom va ichimliklar: *osh, taomlar, somsa, pivo, choy*
- xodimlar: *ofitsiant, ofitsiantlar, xodimlar, oshpaz*
- xizmat: *xizmat, yetkazib berish, buyurtmani olish*
- narx: *narx, narxlar, hisob*
- joy va muhit: *interyer, musiqa, atmosfera, zal, hojatxona*
- muassasaning o'zi: *joy, restoran, kafe* (faqat sharh uni baholasa, A3 ga qarang)

**A1. Faqat nomlangan narsa.** Matnda so'z bilan aytilmagan narsani belgilamang.
*"Juda mazali!"* — termin yo'q: taom nazarda tutilgan, lekin nomlanmagan.

**A2. Bular aspekt emas:**
- olmosh va umumiy so'zlar: *hammasi, hamma narsa, bular, u, barchasi*
- joy ravishlari: *bu yerda, u yerga, shu yerda*
- muallifning o'zi va hamrohlari: *men, oilam, do'stlarim*
- baho so'zlari: *mazali, hamyonbop, ajoyib, zo'r*
- raqam va summalar: *5, 10 dollar, 0, 5 yulduz*
- emojilar: *👌, 👍*
- muassasaning nomi: *"Sulton restorani", "Rayhon"*

**A3. Joy, restoran, kafe.** Sharh muassasani baholasa, belgilang: *"Ajoyib joy"* → *joy* = positive.
Faqat qayerda bo'lganini aytsa, belgilamang: *"Bu restoranda bir necha bor bo'ldim"* → termin yo'q.

**A4. Fikrsiz eslatma ham aspekt.** Nomlangan aspekt haqida fikr bildirilmasa ham belgilang, hissiyoti
`neutral`: *"Uyga yetkazib berish uchun buyurtma qilingan"* → *yetkazib berish* = neutral.

## 2. Chegaralar: qaysi so'zlarni olish kerak

**S1. Matndagidek ko'chiring,** qo'shimchalari va imlo xatolari bilan: *Ofitsantlani, taomlari*.
So'zni lug'at shakliga keltirmang, xatoni tuzatmang. Boshidagi va oxiridagi tinish belgilari va
qo'shtirnoqlarni olmang.

**S2. Turini bildiruvchi so'zlarni qoldiring:** *milliy taomlar, hind taomlari, qalampirli biftek,
sezar salati, bolalar maydoni, kechki ovqat, yetkazib berish, xizmat ko'rsatish*.

**S3. Baho, miqdor va joyni bildiruvchi so'zlarni olib tashlang:** *mazali, chiroyli, katta, uchta,
2 ta, 2-qavatdagi, bu, shu, barcha*. Masalan, *"barcha taomlar"* → *taomlar*, *"uchta taom"* → *taom*.

**S4. Qaratqich (-ning).** Aspektni aniqlashtirsa, qoldiring: *qo'zining oyog'i, salatlarning
porsiyalari*. Muassasa yoki olmosh bo'lsa, olib tashlang: *restoranning oshxonasi* → *oshxonasi*,
*kafening joylashuvi* → *joylashuvi*, *ularning xizmati* → *xizmati*.

**S5. Ro'yxatdagi har bir narsa alohida termin:** *"osh, somsa va choy"* → *osh*, *somsa*, *choy*,
har birining o'z hissiyoti bilan.

**S6. Bir narsa uchun bitta termin.** Ikki nomzod bir narsani nomlasa (*buyurtmani olish* va
*buyurtmani*), faqat qoidalarga mosiga `yes` deng (kerak bo'lsa chegarasini tuzating), qolganlariga `no`.

## 3. Hissiyot (polarity)

- `positive` — ijobiy; `negative` — salbiy; `neutral` — fikr bildirilmagan (faqat eslatilgan, savol yoki reja).
- **Ma'noga qarang, so'zga emas:** *"gap yo'q"* (a'lo) → positive; *"yomon emas"* → positive;
  *"xizmat qilish dahshatli"* → negative; *"2 soat ichida yetkazib berishadimi?"* → neutral.
- Bir aspekt haqida ham yaxshi, ham yomon gapirilsa, muallif ko'proq urg'u bergan tomonni tanlang.
  Ikkalasi teng bo'lsa, `neutral`.

## 4. Jadvalni to'ldirish

Har bir nomzod qatorida:

| ustun | nima yoziladi |
|---|---|
| `is_aspect` | `yes`: bu so'z shu sharhda aspekt termini; `no`: emas |
| `polarity` | faqat `yes` bo'lsa: `positive` / `negative` / `neutral` |
| `corrected_term` | faqat nomzod to'g'ri narsani nomlasa-yu, chegarasi noto'g'ri bo'lsa: to'g'ri so'zlarni sharhdan ko'chiring (*restoranning oshxonasi* → `oshxonasi`) |
| `comment` | ixtiyoriy: ikkilangan bo'lsangiz yozing |

Har bir sharhning **oxirgi qatori** (*"+ aspects missing above"*): ro'yxatda yo'q aspektlarni
`corrected_term` ustuniga shu formatda yozing: `termin = polarity; termin = polarity`
(masalan, `musiqa = positive; narxlar = negative`). Terminni sharhdan ko'chiring. Hech narsa
yetishmasa, bo'sh qoldiring.

**Qilmang:** qatorlarni o'chirmang yoki saralamang; `row_id`, `review` va `candidate_term` ustunlarini
o'zgartirmang. Faylni o'sha nom bilan, `.xlsx` formatida saqlang. Har bir nomzod qatorida `is_aspect`
to'ldirilgan bo'lishi kerak, aks holda skript faylni qabul qilmaydi.

## 5. Mashq (javoblari pastda)

Bu misollar test to'plamidan emas. Avval o'zingiz yeching, keyin javoblar bilan solishtiring.

1. *Ajoyib joy* — nomzodlar: `joy`, `Ajoyib joy`
2. *Juda mazali milliy taom!* — nomzod: `taom`
3. *Hammasi zo'r.* — nomzod: `Hammasi`
4. *Menga bu restoranning oshxonasi juda yoqadi.* — nomzod: `restoranning oshxonasi`
5. *Somsaga gap yo'q, lekin choy sovuq edi.* — nomzod: `Somsaga`
6. *Hozirgina yetkazib berishni buyurdim.* — nomzod yo'q

**Javoblar.**

1. `joy`: yes, positive. `Ajoyib joy`: no (S3, S6).
2. `taom`: yes, positive, corrected_term = `milliy taom` (S2).
3. `Hammasi`: no (A2). Oxirgi qator bo'sh.
4. yes, positive, corrected_term = `oshxonasi` (S4).
5. `Somsaga`: yes, positive ("gap yo'q" = a'lo). Oxirgi qatorga: `choy = negative`.
6. Oxirgi qatorga: `yetkazib berishni = neutral` (A4, S1).
