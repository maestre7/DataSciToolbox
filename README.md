![portada](documentation/logo_dstb.png)

<p align="center">
  <a href="#Introducción">Introducción</a> •
  <a href="#Estructura-repositorio.">Estructura repositorio.</a> •
  <a href="#Variables.">Variables.</a> •
</p>

<h2 id="Introducción"> :pencil: Introducción</h2>

Bienvenido/a a DataSciToolbox

Le damos la bienvenida a la biblioteca de funciones en Python orientadas a machine learning, recopilada por los alumnos de la promoción de abril a agosto de 2023 del bootcamp de Data Science de The Bridge, Digital Talent Accelerator.

En ella, encontrará un extensivo repositorio que abarca todas las fases necesarias para afrontar un proyecto de machine learning, desde el análisis exploratorio de datos hasta la evaluación de los modelos, pasando por las etapas de preprocesamiento, visualización, feature engineering o entrenamiento y evaluación de los modelos.

-   **.github**: contiene los archivos de configuración para GitHub Actions, que se encargan de ejecutar los tests automáticamente cada vez que se hace un commit o un pull request al repositorio. [Ver contenido](#github)
-   **documentation**: contiene los archivos HTML y CSS que generan la documentación completa de la biblioteca, usando la herramienta Sphinx. Puede acceder a la documentación desde este [enlace](https://datascitoolbox.github.io/documentation/index.html). [Ver contenido](#documentation)
-   **src**: contiene los archivos .py con el código fuente de la biblioteca, organizados en diferentes módulos según la funcionalidad que ofrecen. Cada módulo contiene varias funciones que se pueden importar y usar en los proyectos de machine learning. [Ver contenido](#src)
-   **test**: contiene los archivos .py con los tests unitarios que se han realizado para comprobar el correcto funcionamiento de las funciones de la biblioteca, usando la herramienta pytest. También contiene algunas imágenes y archivos auxiliares que se usan en los tests, así como una carpeta donde se guardan los resultados de los tests. [Ver contenido](#test)
<h3 id="github"> :octocat: .github</h3>

Esta carpeta contiene los siguientes archivos:

-   **workflows**: una subcarpeta que contiene los archivos .yml con las instrucciones para GitHub Actions.
-   **flujo_de_trabajo.yml**: un archivo .yml con el flujo de trabajo para ejecutar los tests.

<h3 id="documentation"> :book: documentation</h3>
Esta carpeta contiene los siguientes archivos:

-   **logo_dstb.png**: un archivo .png con el logo de DataSciToolbox.
-   **index.rst**: un archivo .rst con el índice de la documentación.

<h3 id="src"> :computer: src</h3>
Esta carpeta contiene los siguientes archivos[Ver contenido](#src):

-   **modeling.py**
-   **preprocessing.py**
-   **visualization.py**

<h3 id="test"> :computer: src</h3>

Esta carpeta contiene funciones de test para testar los archivos de 
