import folium
import pandas as pd

def plot_map(dataframe, figsize):
    """Esta función crea un mapa interactivo a partir de un DataFrame.

    Argumentos:
        dataframe (pd.DataFrame): El DataFrame que contiene los datos.
        figsize (tuple): El tamaño de la figura del mapa en píxeles (ancho, alto).
    """

    #Crear el mapa usando folium.Map()
    mapa = folium.Map(location=[dataframe['lat'].mean(), dataframe['lon'].mean()], 
                      zoom_start=10, tiles='Stamen Terrain')

    #Añadir marcadores con folium.Marker()
    for i, row in dataframe.iterrows():
        folium.Marker(location=[row['lat'], row['lon']],
                      popup=row['name'], 
                      icon=folium.Icon(color=row['color'])).add_to(mapa)

    #Mostrar el mapa usando folium.Figure()
    fig = folium.Figure(width=figsize[0], height=figsize[1])
    fig.add_child(mapa)
    mapa.save("map.html") #PARA QUE SE VEA EL MAPA
    fig