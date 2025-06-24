from resnet50 import bdRecolector_5c, model_basic, model_preentrenado, bdrecolector_2by2, ResNet50_Sparce

numeroDeEpochs = input("Agregue el número de vueltas que quiere entrenar de cada 20 en 20 epochs el modelo ")

def testFunctions(X,y,clase):
    
    print("Probar basic_model ingrese 1")
    print("probar model_preentrenado ingrese 2")
    print("bdrecolector_2by2 ingrese 3")
    pregunta =input("desea probar algún modelo")
    pregunta= int(pregunta)
    if pregunta ==1:
        model_basic(X,y,clase)
    elif pregunta ==2:
        model_preentrenado(X,y, clase, iteration=1)
    elif pregunta ==3:
        bdrecolector_2by2(clase)
    else:
        print("No hay esa opción, por lo que se procede al entrenamiento general") 


clases = ['Baetidae', 'Canidae', 'Heptageniidae', 'Leptohyphidae', 'Leptophlebiidae']
#clases = ['Caenidae', 'Heptageniidae', 'Leptohyphidae', 'Leptophlebiidae']
print("Para entrenar pares de clases, de 1 contra todos elija 1")
print("Para entrenar las 5 clases elija 2")
print("Para detener presione cualquier otro número")
theChoosen = input()

while(5<6):
    if int(theChoosen) == 1:
        for clase in clases:
            print(clase)
            X,y= bdrecolector_2by2(clase)
            #print(X[0])
            i=0
            testFunctions(X,y,clase)
            while(i < int(numeroDeEpochs)):
                if(i ==0):
                    model_basic(X,y, clase)
                else:
                    model_preentrenado(X,y, clase, i)
                i+=1
    elif int(theChoosen) == 2:
        for clase in clases:
            print(clase)
            X,y= bdRecolector_5c(clase)
            #print(X[0])
            i=0
            while (i < numeroDeEpochs):
                ResNet50_Sparce(X,y,i)

