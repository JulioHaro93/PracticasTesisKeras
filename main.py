from resnet50 import bdRecolector_5c, model_basic, model_preentrenado, bdrecolector_2by2, ResNet50_Sparce, testFunctions, ResNet50_Sparce_kfold, Inception_v3_Sparce
from sklearn.model_selection import KFold
from grad_cam import grad_cam_main
from grapher import grapher, createCSV, lectorCSV


clases = [ 'Heptageniidae', 'Leptohyphidae', 'Leptophlebiidae','Baetidae', 'Caenidae']

print("Para entrenar pares de clases, de 1 contra todos elija 1")
print("Para entrenar las 5 clases elija 2")
print("para probar un mapa de calor grad_cam sobre alguna image, presione 3")
print("Para detener presione cualquier otro número")

theChoosen = input()
final = False

while(final == False):
    numeroDeEpochs = input("Agregue el número de vueltas que quiere entrenar de cada 20 en 20 epochs el modelo ")
    if int(theChoosen) == 1:
        for clase in clases:
            print(clase)
            X,y= bdrecolector_2by2(clase)
            i=0
            while(i < int(numeroDeEpochs)):
                if(i ==0):
                    model_basic(X,y, clase)
                else:
                    model_preentrenado(X,y, clase, i)
                i+=1
        final = True
    elif int(theChoosen) == 2:
        print("Sólo esta red voy a implementarla con k-fold por la capacidad de cómputo del ordenador")
        for clase in clases:
            print(clase)
            X,y= bdRecolector_5c()
            """
                        kf = KFold(n_splits=5, random_state= None, shuffle=False)
            for a, (train_index, test_index) in enumerate(kf.split(X)):
                X_train, y_train = X[train_index], y[train_index]
                X_test, y_test = X[test_index], y[test_index]
            """
            for i in range(int(numeroDeEpochs)):
                if i == 0:
                    Inception_v3_Sparce(X, y, i, clases)
                    print("////////////////////////////////////\nCOMIENZA ENTRENAMIENTO DE RESNET50\n////////////////////////////////////")
                    ResNet50_Sparce(X, y, i, clases)
                    print("////////////////////////////////////\nCOMIENZA ENTRENAMIENTO DE RESNET50\n////////////////////////////////////")

                    #ResNet50_Sparce_kfold(X_train, y_train, X_test, y_test, a, i, clases)
                    
                    print("Entrenamiento, Primera iteración")
                    print(i)
                else:
                    #ResNet50_Sparce_kfold(X_train, y_train, X_test, y_test, a, i, clases)
                    print("Entrenamiento de la clase iteración:")
                    print(i)
                    print("////////////////////////////////////\nCOMIENZA ENTRENAMIENTO DE RESNET50\n////////////////////////////////////")
                    ResNet50_Sparce(X,y, i, clases)
                    print("////////////////////////////////////\nTERMINA ENTRENAMIENTO DE RESNET50\n////////////////////////////////////")
                    print("////////////////////////////////////\nCOMIENZA ENTRENAMIENTO DE INCEPTIONV3\n////////////////////////////////////")
                    Inception_v3_Sparce(X, y, i, clases)  
                    print("////////////////////////////////////\nTERMINA ENTRENAMIENTO DE INCEPTIONV3\n////////////////////////////////////")
        final = True
    elif(int(theChoosen)==3):
        beatidae = 'C:/Users/Julio/Documents/tesis/Tesis-BD/Ephemeroptera/Beatidae/recpadding/5.jpg'
        model_path = 'C:/Users/Julio/Documents/Testing para la Tesis/Results 17-06-2025/CNN-VAE_Curso/ResNet50_fold0_iter1.h5'
        grad_cam_main(img_path=beatidae, model_path=model_path)
    else:
        break

graph = input("Para graficar resultados escriba si")

if graph== 'si' or graph =="SI":
    createCSV()
    df = lectorCSV()
    print(df.head())
    grapher(df)
