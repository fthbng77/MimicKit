# MimicKit boks projesi

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

Reference control artık varsayılan açık. Saldırı dışı durumlarda karakter guard
eklem hedefinde tutulur. Saldırılar sırayla iki varyanta ayrılır: kısa jab ve
daha uzun cross. `fighter_attack_variant` ile `opponent_attack_variant` alanları
hangi varyantın çalıştığını raporlar. Saldırıdan sonra state machine recovery
ve cooldown uygular; aynı yumruk karesi her simülasyon adımında tekrarlanmaz.

Savunma katmanı eklendi: rakip saldırıdayken karakter `BLOCK` durumuna geçer,
guard pozu korunur ve gelen hasar `block_damage_scale` ile azaltılır. Tehdit
bitince karakter tekrar karar döngüsüne döner.
