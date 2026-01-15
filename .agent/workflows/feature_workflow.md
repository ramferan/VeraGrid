---
description: Flujo estándar para añadir una nueva funcionalidad
---

1. Asegúrate de estar en `devel` y actualizado:
   ```powershell
   git checkout devel
   git pull origin devel
   ```

2. Crea una nueva rama para la funcionalidad:
   ```powershell
   git checkout -b feat/nombre-funcionalidad
   ```
   *(Sustituye nombre-funcionalidad por algo descriptivo, ej: feat/configuracion-env)*

3. [TU TRABAJO AQUÍ] 
   - Realiza los cambios necesarios en el código.
   - Crea tests si es posible.

4. Verifica los cambios (tests/linting).

5. Añade los cambios al stage:
   ```powershell
   git add .
   ```

6. Realiza un commit descriptivo:
   ```powershell
   git commit -m "feat: descripción breve de lo que se ha hecho"
   ```

7. (Opcional) Si has terminado, sube la rama o mergea a devel:
   ```powershell
   git push origin feat/nombre-funcionalidad
   ```
