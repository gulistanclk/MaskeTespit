from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import cv2


#Maske tahminini yapan fonksiyon
def detect_and_predict_mask(frame, faceNet, maskNet):
    # Görüntü karesinin boyutlarının alınması
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # opencv modeli ile yüz tespitinin yapılması
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # yüzlerin, yüz konumlarının ve olasılıklarının listelerinin oluşturulması
    faces = []
    locs = []
    preds = []

    # Tespit edilen yüzler üzerinde tek tek gezilerek alttaki işlemler yapılır
    for i in range(0, detections.shape[2]):

        # Tespit ile ilgili güvenin çıkarılması
        confidence = detections[0, 0, i, 2]

        # Güvenilirliği minimum değerden küçük olan tespitlerin filtrelenmesi
        if confidence > 0.5:
            # yüzün x,y koordinatlarının hesaplanması
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Yüzün etrafını çizen kutuların ekran içerisinde kalmasını sağlayan kısım

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # Yüzün ilgili bölümlerinin BGR dan RGB ye çevrilmesi ve 224x224 haline getirilmesi
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # Yüzleri ve yüz konum bilgilerinin atıldığı listeler
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # en az bir yüz tespit edildiyse aşağıdaki işlemler yapılır
    if len(faces) > 0:
        # Eğittiğimiz model ve bulduğumuz yüzler kullanılarak tahminlerin yapıldığı kısım
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)
    # yüz konumları ve tahminler ana fonksiyona döndürülür
    return (locs, preds)


# Yüz tanıma modellerinin diskten alınması. Yüz tanıma için opencv'nin hazır fonksiyonu kullanıldı
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Bizim yaptığımız eğitim sonucunda elde edilen model
maskNet = load_model("mask_detector.model")


print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

#Ana fonksiyon
def detect_mask():

    # Çıkış için q tuşuna basılana kadar video kaynağından görüntü karesi almaya devam eder

    while True:
        total_counter = mask_counter = no_mask_counter = 0
        frame = vs.read()

        # Open cv'nin yüz tanıma modelini ve bizim eğittiğimiz maske tespiti modelini tahmin fonksiyonuna gönderiyoruz
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)


        for (box, pred) in zip(locs, preds):
            #Bulunan yüzün etrafını çizmek için sınırların belirlenmesi
            (startX, startY, endX, endY) = box

            #Maske tespit fonksiyonu dönen tahmin sonuçları (mask + withoutmask = 1)
            (mask, withoutMask) = pred

            #Eğer tahmin sonuçlarından dönen maskeli sonucu, maskesiz sonucundan büyükse Maskeli olarak etiketle
            if mask > withoutMask:
                label = "Maskeli"
                mask_counter += 1

            else:
                label = "Maskesiz"
                no_mask_counter += 1

            #Kayıt başladığından itibaren görülen maskeli ve maskesiz kişilerin sayısı
            total_counter += 1
            print(f"Maskeli : {mask_counter}  -- Maskesiz : {no_mask_counter}  -- Toplam : {total_counter}")
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            #Olasılıkların ekrana yhazdırılması
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # Ekrana getirilecek çerçeve içerisine oluşturulan karenin ve etiketin eklenmesi
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        #Oluşturulan görüntü çerçevesinin(frame) ekrana getirilmesi
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # q tuşuna basınca döngüden çıkılır
        if key == ord("q"):
            break

    # program sonlandırılır
    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    detect_mask()
