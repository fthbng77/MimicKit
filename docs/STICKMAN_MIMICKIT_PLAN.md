# MimicKit boks projesi

## Güncel uygulama ve sınırlar

Son dövüş dilimi: stamina yeterliyken her ikinci saldırı iki vuruşluk jab–cross
kombinasyonudur. İkinci vuruşun isabet hakkı ayrı sıfırlanır; iki vuruş sonrası
recovery zorunludur. Test iki karakterin de kombinasyonun ikinci vuruşuna
ulaştığını denetler. Episode 60 saniye; görüntü yakın kadraj ve can/durum/stamina
bilgisiyle kaydedilir. Çalıştırma: `python tools/combat_state_smoke.py --visualize`.

- İki elli guard, ayrı sol/sağ düz yumruk ve yumuşak guard dönüşü var.
- Cross sırasında küçük gövde dönüşü var; tam ağırlık aktarımı henüz yok.
- BLOCK, SLIP ve BACKSTEP kararları dönüşümlü uygulanıyor.
- Stamina saldırıda 0.25 azalır, dinlenmede saniyede 0.12 yenilenir;
  0.25 altında yeni saldırı başlamaz.
- Hareket sırasında alternatif bacak animasyonu var; ayak temasını kilitleyen
  IK ve fiziksel yürüyüş henüz tamamlanmadı.
- Yaklaşma ve geri çekilmede kök hızı artık 0.65 saniyelik adım fazında
  uygulanır; ayak basışında duraklar. Bu kinematik adım kaymayı azaltır,
  zemin temasını fiziksel olarak sabitlemez.
- Ayak basışında seçilen destek ayağının dünya koordinatı sabitlenir.
  `support_raw_slip` düzeltme öncesi, `support_foot_slip` düzeltme sonrası
  kaymayı raporlar. 900 adımlık testte 289 destek örneği: en yüksek
  0.0162 m ham kayma, düzeltme sonrası 0.0000 m. Bu kinematik düzeltmedir;
  Newton temas kuvveti veya tam bacak IK çözümü değildir.
- Boks hasarı yalnız ellerden gelir. Blokta ölüm de knockdown akışına girer.
- `tools/boxing_pose_test.py` kol ayrımını, dönüşü ve dirsek yüksekliğini denetler.
- Combat testi 900 adım, iki taraflı hasar, savunma, kontrollü hit-stun,
  ring, knockdown ve reset kontrollerini kapsar.

Demo pozu doğrudan uygulandığı için residual PPO fizik kontrolü hâlâ tamamlanmış
değildir. Bu aşamayı çalışan/öğrenilmiş fiziksel denge olarak değerlendirmeyin.
Doğal ayak teması ve residual fizik doğrulaması açık işlerdir.

## Mevcut durum

Combat ortamı Newton üzerinde iki humanoid çalıştırır. Ring, yaklaşma, saldırı
cooldown'ı, recovery, hit-stun, geri itilme, can, knockdown ve iki taraflı hit
akışı hazırdır. `tools/combat_state_smoke.py` 300 adımlık başsız doğrulama yapar.

`BoxingReferenceController` rakip yönü ve mesafeden şu niyetleri üretir:

```text
GUARD / APPROACH / JAB / RECOVER / RETREAT
```

`reference_control` açıldığında referans eklem hedefi ile residual action
birleştirilir. Varsayılan combat demosu korunur.

## Sıradaki uygulama sırası

1. Guard pozu
   - Omuz, dirsek ve elleri koruma konumuna getir.
   - Rakibe dönük gövde yönünü koru.

2. Jab / cross
   - Sol ve sağ omuz-dirsek hedeflerini ayır.
   - Saldırı aktif penceresini eklem hedefiyle eşleştir.
   - Yumruk sonrası guard'a dön.

3. Ayak çalışması
   - İleri, geri, lateral adım ve pivot.
   - Ring kenarında merkeze dön.
   - İdeal mesafeyi koru.

4. Savunma
   - Guard, block, slip ve backstep.
   - Savunma sırasında hasar azalt.
   - Hit algılandığında saldırıyı kes.

5. Residual PPO
   - Referans eylem temel hareketi üretir.
   - PPO yalnız denge, mesafe, zamanlama ve darbe düzeltir.
   - `final_action = reference_action + residual`.

6. Tempo
   - Stamina.
   - Yumruk ve recovery maliyeti.
   - Kombinasyon sınırı.

## Doğrulama

- 300 adım combat smoke.
- Guard pozu görsel test.
- Sol jab / sağ cross görsel test.
- İki taraflı hasar ve knockdown.
- Ring dışına çıkmama.
- Aynı anda sürekli saldırı olmaması.
- `fighter_state`, `opponent_state`, mesafe ve cooldown bilgilerinin çıktıda
  görünmesi.

## Çalıştırma

```bash
.venv-py310/bin/python tools/combat_state_smoke.py
```

Ana dosyalar:

- `mimickit/envs/combat_env.py`
- `mimickit/envs/boxing_controller.py`
- `data/envs/combat_humanoid_env.yaml`
- `tools/combat_state_smoke.py`

## Son görünür dövüş davranışı

Güncel prototip: reference modu iki kollu guard ile ayrı sol/sağ kol
yörüngeleri üretir (0: sol jab, 1: sağ düz yumruk). Süreler 0.9/1.2 saniye;
yumuşak interpolasyon sonunda hedef tam guard'a döner. Kayıtlı klibin kök
hareketi bu modda uygulanmaz. Menzil 0.65 metreye ayarlanmıştır.
Bu hâlâ kinematik demo: residual motor komutu pozu yöneten kontrol değildir.
Cross için gövde/kalça ağırlık aktarımı ve doğal guard kalibrasyonu henüz eksiktir.

Reference control artık varsayılan açık. Saldırı dışı durumlarda karakter guard
eklem hedefinde tutulur. Saldırılar sırayla iki varyanta ayrılır: kısa jab ve
daha uzun cross. `fighter_attack_variant` ile `opponent_attack_variant` alanları
hangi varyantın çalıştığını raporlar. Saldırıdan sonra state machine recovery
ve cooldown uygular; aynı yumruk karesi her simülasyon adımında tekrarlanmaz.

Savunma katmanı eklendi: rakip saldırıdayken karakter `BLOCK` durumuna geçer,
guard pozu korunur ve gelen hasar `block_damage_scale` ile azaltılır. Tehdit
bitince karakter tekrar karar döngüsüne döner.
