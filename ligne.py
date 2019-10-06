import cv2
import numpy as np
import matplotlib.pyplot as plt



def creer_coordonees(image, ligne):
    pente, ordonne_origine = ligne
    y1 = int(image.shape[0])
    y2 = int(y1*3/5)
    x1 = int((y1 - ordonne_origine)/pente)
    x2 = int((y2 - ordonne_origine)/pente)
    return [[x1, y1, x2, y2]]






def moyenne_pente_origine(image,lignes):
    ajustement_gauche    = []
    ajustement_droit   = []
    # S'assurer que les lignes ne sont pas vides (sont bien detectees)
    if lignes is None:
        return None
    for ligne in lignes:
            x1, y1, x2, y2 = ligne.reshape(4)
            parametre = np.polyfit((x1,x2), (y1,y2), 1)
            pente = parametre[0]
            ordonne_origine = parametre[1]
            if pente < 0:
                ajustement_gauche.append((pente, ordonne_origine))
            else:
                ajustement_droit.append((pente, ordonne_origine))

    moyenne_ajustement_gauche  = np.average(ajustement_gauche, axis=0)
    moyenne_ajustement_droit = np.average(ajustement_droit, axis=0)
    ligne_gauche  = creer_coordonees(image, moyenne_ajustement_gauche)
    ligne_droite = creer_coordonees(image, moyenne_ajustement_droit)


    return np.array([ligne_gauche, ligne_droite])



def canny(img):
    gris = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #changer l'image de color à gris
    kernel = 5
    flou = cv2.GaussianBlur(gris, (kernel,kernel), 0) # réduire le bruit de l'image pour ne pas affecter la detection des bords
    canny = cv2.Canny(flou, 50, 150)
    return canny


def region_of_interest(image):
    hauteur = image.shape[0]
    largeur = image.shape[1]
    mask = np.zeros_like(image)

    triangle = np.array([[
    (200,hauteur),
    (550,250),
    (1100, hauteur),
    ]], np.int32)

    cv2.fillPoly(mask, triangle, 255)
    image_masquee = cv2.bitwise_and(image,mask)
    return image_masquee

def afficher_lignes(image,lignes):
    traits_image = np.zeros_like(image)
    if lignes is not None:
        for ligne in lignes:
            x1,y1,x2,y2 = ligne.reshape(4)
            cv2.line(traits_image,(x1,y1),(x2,y2),(255,0,0),10)

    return traits_image

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    canny_image = canny(frame)
    image_coupee = region_of_interest(canny_image)
    lignes = cv2.HoughLinesP(image_coupee, 2, np.pi/180, 100, np.array([]), minLineLength=40,maxLineGap=5)
    lignes_moyenne = moyenne_pente_origine(frame, lignes)
    traits_image = afficher_lignes(frame, lignes_moyenne)
    trace_image = cv2.addWeighted(frame, 0.8, traits_image, 1, 0)
    cv2.imshow('resultat', trace_image)

    if cv2.waitKey(1)  & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
