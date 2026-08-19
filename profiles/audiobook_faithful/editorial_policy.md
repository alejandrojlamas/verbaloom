# Audiolibro - traduccion fiel

Perfil reutilizable para traducir libros completos con fidelidad y preparar una version limpia para escucha.

## Objetivo

- La traduccion debe ser fiel al original: no resumir, no censurar, no inventar y no cambiar datos.
- La salida principal conserva el flujo normal de la aplicacion.
- Cuando este perfil esta activo, el sistema genera ademas un companion de audiolibro con texto narrable.
- La sanitizacion de audiolibro elimina paginacion, marcas de agua, links y llamadas de nota del cuerpo principal.
- Las notas, referencias y creditos visuales se mueven a un apendice para no interrumpir la escucha.
- Los pies de imagen informativos se integran como descripcion breve cuando aportan contenido real.

## No hacer

- No convertir captions en explicaciones inventadas.
- No describir una imagen si el texto fuente no aporta esa informacion.
- No eliminar contenido sustantivo: si no debe interrumpir la escucha, moverlo al apendice.
- No aplicar reglas de este perfil a otros perfiles.
