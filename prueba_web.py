from distutils.command import upload
import streamlit as st
import pickle
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
#import random
#from random import sample


def main():
    st.title('Rompecabezas extremo')
    st.sidebar.header('Parametros')

    #funcion para poner los parametrso en el sidebar
    def user_input_parameters():
        part_filas = st.sidebar.slider('Filas', 2,4,10)
        part_columnas = st.sidebar.slider('Columnas', 2,4,10)
        data = {'Particion_Filas': part_filas,
                'Particion_Columnas': part_columnas
                }
        features = pd.DataFrame(data, index=[0])
        return features

    df = user_input_parameters()

    st.subheader("RompecabezasApp es una WebApp con la cual podras elegir dividir una imagen acorde a los parametros seleccionados , para luego fusionarlos de forma aleatoria")
    #st.write(df)

    
    uploaded_file = st.file_uploader("Elija la imagen",type="jpg")

    if uploaded_file is not None:
        # Convert the file to an opencv image.
        print("Nombre del archivo : ",str(uploaded_file))
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        #st.image(opencv_image, channels="RGB")
        part_list = [[0 for i in range(df['Particion_Columnas'][0])] for j in range(df['Particion_Filas'][0])]
        #part_list = [['' for i in range(3)] for j in range(3)]


        cant_colum = df['Particion_Columnas'][0]
        cant_filas = df['Particion_Filas'][0]
        dim = np.shape(opencv_image)
        #x =str(dim)+"          "
        #y =str(dim)+"          "
        list_img = []
        cont=0
        for i in range(cant_filas):
            for j in range(cant_colum):
                part_list[i][j] = opencv_image[ int(i*dim[0]/cant_filas):int((i+1)*dim[0]/cant_filas),int(j*dim[1]/cant_colum):int((j+1)*dim[1]/cant_colum) ] 
                list_img.append(opencv_image[ int(i*dim[0]/cant_filas):int((i+1)*dim[0]/cant_filas),int(j*dim[1]/cant_colum):int((j+1)*dim[1]/cant_colum) ] )
                cont=cont+1
                #y = y +"  "+str([ [int(i*dim[0]/cant_filas),int((i+1)*dim[0]/cant_filas)]  , [int(j*dim[1]/cant_colum),int((j+1)*dim[1]/cant_colum)]  ])
                #x = x  +"  "+"partes[{}][{}]".format(i,j)
        #st.subheader(x)
        #st.subheader(cont)

        #list_img_len = []

        #fig = plt.figure(figsize=(20,8*len(list_img)))
        #ax = fig.subplots(1, len(list_img))
        #for w in range(len(list_img)):
        #    ax[w].set_title("Figura {}".format(w+1))
        #    ax[w].imshow(list_img[w],cmap='gray')
        #    list_img_len.append(np.shape(list_img[w]))
        #    ax[w].axis("off")

        #st.pyplot(fig)
        
        #res = set()  
        #temp = [res.add((a, b)) for (a, b) in list_img_len  if (a, b) and (b, a) not in res] 
        #st.subheader(list(res))

        
        dic_img = { i:list_img[i] for i in range(len(list_img)) }
        
        img_final = np.array([])
        img_final_part = []

        img_final_total = np.array([])

        #fig1 = plt.figure(figsize=(4*cant_filas,4))
        #ax1 = fig1.subplots(1, cant_filas)
        print("cant_filas :" ,cant_filas)
        print("cant_columnas :" ,cant_colum)
        for i in range(cant_filas):
            for j in range(cant_colum):
                dim_temp = np.shape(part_list[i][j])
                parte= np.random.choice(list(dic_img.keys()))
                dim_temp_part = np.shape(dic_img[parte])
                
                
                while( not(dim_temp_part == dim_temp) ):
                    parte= np.random.choice(list(dic_img.keys()))
                    dim_temp_part = np.shape(dic_img[parte])
                    
                if (j==0):
                    #print("estoy en el if x2",dim_temp_part)
                    img_final = np.zeros(dim_temp_part)
                    tempo = dim_temp_part
                img_final = np.hstack([img_final,dic_img[parte]])
                del dic_img[parte]


            #ax1[i].set_title(str(np.shape(img_final[:,tempo[1]:])))
            #ax1[i].imshow(img_final[:,tempo[1]:],cmap='gray')
            img_final_part.append(img_final[:,tempo[1]:])

            if (i==0):
                dim_temp_part_2 = np.shape(img_final[:,tempo[1]:])
                img_final_total = np.zeros(dim_temp_part_2)
                tempo_2 = dim_temp_part_2
            
            img_final_total = np.vstack( [ img_final_total , img_final[:,tempo[1]:] ] )



            #print(">>>>>>>>>>hola mundo",np.shape(img_final),type(img_final) ,tempo) 

        
        imagen_resultado = np.array(img_final_total[tempo_2[0]:,:,:] ,np.uint8 )
        #print(np.shape(imagen_resultado) , tempo_2)
        #st.pyplot(fig1)

        fig2 = plt.figure(figsize=(20,20))
        ax = fig2.subplots(1,2)
        ax[0].imshow(opencv_image,cmap='gray')
        ax[0].axis("off")
        ax[1].imshow(imagen_resultado,cmap='gray')
        ax[1].axis("off")
        st.pyplot(fig2)
        
            

if __name__ == '__main__':
    main()

