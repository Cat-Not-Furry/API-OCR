# API OCR para proyecto AIDA #<br>
#### Esta API esta en desarrollo y se planea pueda identificar checkboxes y cuadros de texto para desarrollo<br>de documentos, si quieres probarlo, descarga usa:<br> ####

```
export TESSDATA_PREFIX=$(pwd)/tessdata
```

```
./bin/tesseract [tu-imagen].jpg stdout -l spa --psm 6
```

#### O probar en linea con render ####

```
https://render.com/docs/troubleshooting-deploys
```

#### Utiliza el metodo post, sube el archivo y ejecuta ####
