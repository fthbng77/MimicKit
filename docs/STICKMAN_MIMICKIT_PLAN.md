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
