import cv2
import os

# Verifique se o arquivo de vídeo existe no diretório atual
video_path = 'vd02.mp4'  # Substitua pelo caminho do seu vídeo
if not os.path.exists(video_path):
    print(f"O arquivo {video_path} não foi encontrado!")
    exit()

# Carregar o classificador HOG para detecção de pessoas
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Carregar o vídeo
cap = cv2.VideoCapture(video_path)

# Definir a resolução menor para os frames (Reduzir resolução para acelerar)
frame_width = 640
frame_height = 360

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Redimensionar o frame para acelerar o processo (menor resolução)
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Converter para escala de cinza (necessário para o detector)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar pessoas na imagem com parâmetros ajustados para maior velocidade
    boxes, weights = hog.detectMultiScale(
        gray,
        winStride=(8, 8),  # Aumente winStride para maior velocidade
        padding=(8, 8),  # Padding maior pode ajudar na velocidade
        scale=1.1  # Maior escala para reduzir a precisão e acelerar
    )

    # Para cada pessoa detectada, desenhar uma caixa delimitadora e adicionar texto
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Exibir a mensagem "Invasor detectado" acima da caixa delimitadora
        cv2.putText(frame, 'Invasor detectado', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    # Exibir a imagem com a detecção e a mensagem
    cv2.imshow('Detecção de Pessoa', frame)

    # Pressionar 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar o vídeo e fechar as janelas
cap.release()
cv2.destroyAllWindows()

