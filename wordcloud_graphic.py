import matplotlib.pyplot as plt
from wordcloud import WordCloud

def plot_wordcloud(text, figsize, filename):
    """Esta función crea una nube de palabras a partir de un texto.

    Argumentos=
        text (str): El texto que contiene las palabras.
        figsize (tuple): El tamaño de la figura de la nube de palabras en pulgadas (ancho, alto).
        filename (str): El nombre del archivo en el que se guardará la figura.
    """

    #Nube de palabras creada usando WordCloud()
    wc = WordCloud(background_color='white', 
                   max_words=100, 
                   width=figsize[0]*100, 
                   height=figsize[1]*100).generate(text)

    #Tamaño de la figura
    plt.figure(figsize=figsize)

    #Mostrar la nube de palabras usando plt.imshow() y plt.axis()
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')

    #Guardar
    plt.savefig(filename)

    #Mostrar
    plt.show()
