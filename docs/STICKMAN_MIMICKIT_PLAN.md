# Çubuk adam — MimicKit geliştirme planı

## Karar

Ana öğrenme ortamı MimicKit olacak. ROS 2 ve Gazebo kaynakları bu depoda
değiştirilmeyecek; ileride politika gösterimi veya robotik doğrulama gerekirse
ayrı proje olarak kullanılacak.

MimicKit'in test edilmiş Newton `1.0.0` arka ucunu, bu depo içindeki izole bir
Python ortamında kullanacağız. Bu seçim, Isaac Sim / Isaac Lab'in tam
renderleme hattının önerdiği 16 GB VRAM gereksiniminden kaçınır. Eğitim
başlangıçta başsız, değerlendirme ise görüntülü çalışır.

## Aşama 0 — Temel doğrulama

Başarı ölçütü:

1. İzole ortamda Newton ve MimicKit import edilir.
2. GPU ile Newton'un küçük bir örneği çalışır.
3. MimicKit'in resmî insan modelinin hazır hareket klibi görüntülenir.

Bu aşamada özel karakter, eğitim veya büyük veri indirme yoktur.

### Uygulama durumu

İlk iki karakterli ortam iskeleti eklendi:

- `mimickit/envs/combat_env.py`: aynı fizik ortamında savaşçı ve pasif rakip.
- `data/envs/combat_humanoid_env.yaml`: başlangıç pozları ve karakter ayarları.
- `args/combat_humanoid_args.txt`: test çalıştırma argümanları.

Bu dilimde yumruk hasarı, can sistemi ve self-play henüz yoktur. Önce Newton
çalışma bağımlılıkları doğrulanmalı, ardından temas ve darbe ölçümü eklenmelidir.

İkinci uygulama diliminde geometrik darbe algılama eklendi. Saldırı el/ayakları
ile rakibin baş, gövde ve pelvis bölgeleri arasındaki mesafe ölçülüyor; isabet
durumunda bekleme süresi, hasar, ödül ve rakip sağlık değeri güncelleniyor.
Fizik temas sensörü ve saldırı animasyonu henüz sonraki dilimdedir.

Üçüncü uygulama diliminde `humanoid_punch.pkl` ile tekrarlanabilir yumruk
demonstrasyonu eklendi. Demo modu açıkken mavi karakter hareket klibini oynatır,
rakibe doğru yönelir ve isabet algılandığında kırmızı karakterin canı azalır.
Bu aşama henüz öğrenen politika veya oyuncu kontrolü değildir; görsel ve fizik
akışını doğrulayan kontrollü bir testtir.

Dördüncü uygulama diliminde saldırı penceresi ve darbe tepkisi eklendi. Yumruk
yalnızca aktif zaman aralığında hasar verebilir ve bir yumruk döngüsünde en fazla
bir hit üretebilir. İsabet sonrasında rakibin canı azalır, kısa süreli sendeleme
durumu oluşur ve rakibe geriye doğru hız uygulanır.

Bu darbe/can dilimi tamamlandı. Combat ortamı artık saldırı ve hedef vücut
bölgelerini, relatif kök konum/hızını, el/ayak hareket hızını, temas kuvvetini,
aktif saldırı penceresini, hasarı, iki tarafın can alanlarını ve knockdown
durumunu raporluyor. Hit hesabı mesafe, hareket/ yaklaşma hızı ve mevcut temas
kuvveti sinyallerini birlikte değerlendiriyor.

Sonraki dilimde rakip için aynı yumruk hareketinin ters yönde oynatılması eklendi.
Artık iki karakter de saldırı penceresinde isabet üretebiliyor; rakip isabeti
`fighter_health` değerini azaltıyor ve bu değer sıfıra indiğinde bölüm FAIL ile
sonlanıyor. Böylece tek taraflı kontrollü demo yerine iki taraflı hasar akışı
başlangıç seviyesinde doğrulanmış oldu. Henüz öğrenen rakip politikası veya
oyuncu kontrolü değildir; saldırı davranışı tekrarlanabilir demo hareketidir.

## Aşama 1 — Resmî örneği anlamak

Başarı ölçütü:

1. Tek hazır hareket klibi görüntülenir.
2. Önceden eğitilmiş bir hareket politikası küçük ortam sayısıyla test edilir.
3. Başlangıç pozu, gözlem, eylem, ödül ve video kaydı dosyaları belgelenir.

Amaç başarıyla yürüyen yeni bir model üretmek değil, proje akışını kanıtlamaktır.

## Aşama 2 — Dört eklemli denge karakteri

Başarı ölçütü:

1. Tam görünümlü, ancak yalnız sağ/sol kalça ve dizleri aktif olan özel model.
2. Diğer eklemler güvenli başlangıç pozunda tutulur.
3. Politika 3 saniye düşmeden kalmayı öğrenir.
4. Her değerlendirmede kısa video, ödül grafiği ve checkpoint üretilir.

## Aşama 3 — Müfredat

`denge → küçük itme → ayak bileği ekleme → adım → yürüyüş → gard → tek hareket`

Yeni eklem veya görev, önceki başarı ölçütü düzenli sağlandıktan sonra eklenir.

## Şimdilik kapsam dışı

- Gazebo için MimicKit motor adaptörü yazmak.
- 4096 paralel ortamla eğitim.
- İki ajanlı dövüş ve self-play.
- Referans hareketten yumruk / tekme taklidi.

Bu maddeler temel denge politikası kanıtlandıktan sonra ayrı kararlarla ele
alınacaktır.

## Durum tabanlı dövüş demosu

`CombatState` kodları: `IDLE=0`, `APPROACH=1`, `ATTACK=2`, `RECOVER=3`,
`RETREAT=4`, `HIT_STUN=5`, `DOWN=6`. Her ortamda iki savaşçının durum,
durum süresi, saldırı cooldown ve toparlanma süreleri ayrı tutulur.

Demo kontrolü rakibe döner, 1.15 m mesafeye yaklaşır, 1.5 saniyelik saldırıdan
sonra 1.0–1.5 saniye toparlanır. Yeni saldırının cooldown süresi, saldırı bitişinden
itibaren 1.5–3.0 saniyedir. Başlangıç cooldown değerleri farklıdır; aynı anda
saldırı başlangıcı engellenir. Darbe saldırıyı keser, sarı renk ve hit-stun
başlatır; ardından geri çekilme gelir. Klipteki ileri hamle korunur, yaklaşma
konumu her döngüde başlangıca sıfırlanmaz. Ring kenarında hareket merkeze yönelir;
sınırdaki dışarı hız sıfırlanır. Can sıfırlandığında kaybeden DOWN, kazanan IDLE
olur; saldırılar durur, bölüm başarı/başarısızlık ile tamamlanır.

`info` alanları: `fighter_state`, `opponent_state`,
`fighter_attack_cooldown`, `opponent_attack_cooldown`, `fighter_hit_stun`,
`opponent_hit_stun`, `fighter_distance`, `combat_state_names`.
Durum kodlarının adları son alandan okunabilir. Politika gözlem tensörünün boyutu
korunur; yeni tanılama alanları `info` üzerinden sunulur.

Doğrulama (depo kökünde, Newton/CUDA ortamında):

```bash
.venv-py310/bin/python tools/combat_state_smoke.py
.venv-py310/bin/python tools/combat_state_smoke.py --visualize
```

Test 300 adım boyunca durum geçişlerini, iki taraflı gerçek isabeti, sonlu
gözlemleri ve ring sınırlarını denetler; ayrıca sınır çizgisinde dışarı hız,
iki taraflı knockdown ve reset kontrollerini çalıştırır. Görsel seçenek
`output/combat_humanoid/state_machine.mp4` ve örnek PNG kareleri üretir.
Bu hâlâ hazır klip kullanan kontrollü demodur; öğrenilmiş hareket politikası değildir.

Doğrulama sonucu: başsız ve görsel 300 adımlık koşular geçti (seed 7).
Mavi 2, kırmızı 1 isabet üretti. İki tarafta saldırı/toparlanma ve
hit-stun/geri çekilme geçişleri gözlendi; sınır hızı, knockdown ve reset
kontrolleri geçti. Kaydedilen karelerde iki karakterin darbe sırasında sarıya
dönmesi ve normal rengine dönmesi doğrulandı.

## Ring sahnesi

`boxing_ring.xml` sabit ring varlığı eklendi: 4 × 4 m dövüş alanı,
36 cm yükseltilmiş platform, minder ve kenar çizgileri, koyu apron,
dört köşe direği, kırmızı/mavi/nötr köşe yastıkları, dört sıra halat,
halat bağlantıları ve giriş basamakları. Halatlar ve direkler sabit çarpışma
geometrisidir; esnek halat simülasyonu kullanılmaz.

`ring_floor_height` minder yüzeyini tanımlar. Başlangıç kök yükseklikleri ve
knockdown pozu bu yüzeye göre ayarlanmıştır. Ring varlığı 4 × 4 m alanla
uyumludur; farklı ölçü girildiğinde açık hata verir. Ring merkezi
`ring_bounds` üzerinden konumlandırılır. Başsız/görsel regresyon için
`tools/combat_state_smoke.py` kullanılır.

Newton görsel uyumluluğu: çarpışma şekillerinin yanında ayrı görsel kopyalar
bulunur; böylece dekoratif şekiller eklendiğinde minder ve halatlar gizlenmez.
Motor, açık nesne rengi verilmemiş MJCF varlıklarında adlandırılmış `geom`
öğelerinin `rgba` renklerini görüntüleyiciye aktarır. Ring kamerası sabit,
yükseltilmiş ve geniş açılı kadraj kullanır; platform ile giriş basamakları
birlikte görünür. Başsız ve görsel 300 adımlık ring regresyonları geçti.

## Hazır politika doğrulaması

Mevcut DeepMimic PPO checkpoint'i küçük ortam sayısıyla Newton/CUDA üzerinde
doğrulandı. `data/models/deepmimic_humanoid_spinkick_model.pt` yüklenerek
4 ortamda test çalıştırıldı.

Son çalıştırma sonucu: `Mean Return = 20.4804`, `Mean Episode Length = 34.5`,
`Episodes = 4`. Rastgele reset nedeniyle değerler çalıştırmalar arasında değişir.
Model yükleme ve rollout hatasız tamamlandı. Tekrar çalıştırma:

```bash
.venv-py310/bin/python mimickit/run.py \
  --arg_file args/deepmimic_humanoid_validation_args.txt
```

Bu aşama mevcut checkpoint'in çalıştığını kanıtlar. Yeni dört eklemli denge
karakteri ve fizik tabanlı eğitim sonraki aşamadır; bu aşamada eğitim başlatılmadı.

## Dört eklemli denge başlangıcı

`balance_humanoid_env.yaml` ve `BalanceEnv`, dört bacak eklemini politika
eylemine açar: iki kalça ve iki diz. Kalça eklemleri spherical olduğu için bu
dört eklem toplam sekiz DoF üretir. Diğer DoF hedefleri başlangıç pozunda
tutulur. Ödül kök yüksekliği, dikey yön ve aktif eklem hareket cezasını kullanır;
kök yüksekliği veya yön eşiği aşılırsa bölüm FAIL olur. Bu dilim yalnız ortam
ve başsız doğrulama içindir; eğitim checkpoint'i üretmez.

Sıfır eylem başsız testi geçti: 4 ortamda 90 simülasyon adımı, ortalama ödül
`134.5507`, ortalama bölüm uzunluğu `90.0`, dört bölüm. Bu sonuç denge
politikasının öğrenildiğini göstermez; ortam, eylem haritalaması, ödül ve
düşme/timeout sonlandırmasının çalıştığını gösterir.

Eğitim dilimi eklendi: `balance_humanoid_ppo_agent.yaml`, 128 paralel ortam,
PPO ve 81.920 örnek. Reset sırasında aktif DoF'lara küçük konum bozuntusu ve
kök gövdeye en fazla 0.10 rad tilt uygulanır. Bu bozuntu, sıfır eylemle ayakta
durma yerine düzeltici politika öğrenmeyi gerekli kılar.

İlk PPO eğitimi `output/balance_humanoid/model.pt` altında 81.920 örnekte
checkpoint üretti. Devam eğitimi `output/balance_humanoid_continue/model.pt`
altında 524.288 örnek tamamladı. Bağımsız 32 ortamlı testte ortalama bölüm
uzunluğu `83.16/90` adıma, ortalama getiri `121.36` değerine ulaştı. Sabit seed
ile görsel rollout 90/90 adım boyunca ayakta kaldı; son `root_up_z` değeri
`0.9426` idi. Video: `output/balance_humanoid_continue/policy_rollout.mp4`.

## Küçük itme başlangıcı

`balance_push_humanoid_env.yaml`, denge ortamına ilk kez 0.75 saniyede ve
ardından her saniye rastgele yatay yönde 0.20 m/s kök hız darbesi ekler.
`balance_last_push` gözlem bilgisi son uygulanan darbeyi raporlar. Politikanın
gözleminde kök hızı zaten bulunduğu için eylem/gözlem boyutu değiştirilmez.

Push PPO eğitimi `output/balance_push_humanoid/model.pt` altında 524.288
örnekte checkpoint üretti. Bağımsız 32 ortamlı test, bölüm uzunluğunu
`70.52/90` adımdan `83.91/90` adıma çıkardı; ortalama getiri `99.39`dan
`122.20`ye yükseldi. Sabit seed görsel rollout, iki düşük hızlı itmeye rağmen
90/90 adım ayakta kaldı; son `root_up_z=0.9984` idi.

## Ayak bileği ekleme başlangıcı

`balance_ankle_humanoid_env.yaml`, iki ankle eklemini de açar. Kalçalar ve
ankle'lar spherical, dizler hinge olduğu için eylem alanı 14 DoF'tur. Düşük
hızlı itme müfredatı korunur. Önce ortam doğrulanır; 8 DoF push checkpoint'i
farklı eylem boyutu nedeniyle bu aşamaya yüklenmez.

14 DoF ayak bileği PPO eğitimi `output/balance_ankle_humanoid/model.pt`
altında 524.288 örnekte tamamlandı. Bağımsız test sonucu ortalama getiri
`111.93`, ortalama bölüm uzunluğu `76.91/90` adım. Eylem alanı ve eğitim hattı
çalışıyor; bu checkpoint henüz 3 saniyelik düzenli denge hedefini karşılamıyor.

## Adım davranışı başlangıcı

`SteppingEnv`, 14 DoF ayak bilekli denge ortamını ileri hareket hedefiyle genişletir.
Kök x hızı `0.25 m/s` hedeflenir; `0.9 s` adım fazı sinüs/kosinüs olarak gözleme
eklenir. Denge, diklik ve itme ödülleri korunur; ilerleme ödülü hedef hıza
yakınlığa göre eklenir. Ayak izi temas planlayıcısı sonraki aşamadır.

`output/stepping_humanoid/model.pt` checkpoint'i 524.288 örnek PPO eğitimiyle
üretildi. Bağımsız 32 ortamlı testte ortalama getiri `73.24`, ortalama bölüm
uzunluğu `53.78` adım oldu. Görsel rollout `60` adımda düşme ile sonlandı.
Video: `output/stepping_humanoid/policy_rollout.mp4`.

## Fazlı ayak yerleşimi

Stepping gözlemine sağ/sol ayak fazı, hedef ileri adım (`0.16 m`), salınım
yüksekliği (`0.04 m`) ve dört boyutlu ayak hedef hatası eklendi. Ödül, hedef
hıza ek olarak ayak hedefinden sapmayı cezalandırıyor. Yeni politika ayrı
checkpoint'te tutuldu: `output/gait_humanoid/model.pt`.

262.144 örnek PPO eğitimi sonrası bağımsız 32 ortam testi ortalama getiri
`93.45`, ortalama bölüm uzunluğu `66.25` adım verdi. Görsel rollout `72`
adım sürdü; sonraki aşama temas algısı ve ayak yere basma doğrulaması.

## Ayak temas doğrulaması

`SteppingEnv` artık Newton yer temas kuvvetlerini iki ayak için gözleme taşır.
Destek fazındaki ayak teması ödüllenir; salınım fazındaki erken temas azaltılır.
`foot_contact` ve `foot_target_error` bilgi alanları gözlenebilir. Gözlem boyutu
`133` oldu.

Temas müfredatı PPO eğitimi `output/contact_gait_humanoid/model.pt` altında
`131.072` örnekte tamamlandı. Bağımsız testte ortalama getiri `81.21`, ortalama
bölüm uzunluğu `59.78` adım. Görsel rollout `55` adımda düşme ile sonlandı.

## Boks altyapısı: reference + residual

Yürüyüş politikası ana hedef olmaktan çıkarıldı. Boks için yeni katman
`BoxingReferenceController` oldu. Controller guard, yaklaşma, jab, toparlanma
ve geri çekilme niyetlerini; rakip yönünü, mesafeyi ve guard hedefini üretir.
`CombatEnv` mevcut fizik, yumruk, hasar ve ring sınırlarını korur; yeni referans
alanları `info` çıktısında görünür. Sonraki adımda bu referans eylem, küçük
residual PPO düzeltmesiyle birleştirilecek:

```text
reference boxing motion + residual policy correction -> Newton action
```

300 adımlık combat smoke testi yeni katmanla geçti; iki taraflı vuruş,
hit-stun, ring ve knockdown davranışları korundu.
