## What is the Problem
Cac UI frontend cu (ui1_frontend, ui2_frontend, ui3_frontend_old) dung de chay UI doc lap/legacy, giu de tham chieu va tuong thich.

## Information Context
File chinh:
- pedi_oku_landslide/ui/views/ui/ui1_frontend.py
- pedi_oku_landslide/ui/views/ui/ui2_frontend.py
- pedi_oku_landslide/ui/views/ui/ui3_frontend_old.py
- pedi_oku_landslide/ui/views/ui/ui1_viewer.py
Phu thuoc backend:
- pipeline.runners.ui1_backend/ui2_backend/ui3_backend
Quy uoc/luu y:
- Cac file nay co the import duong dan tuong doi (pipeline.* hoac ui1_frontend) va co print warning neu import fail.
- Duong dan output van theo output/<project>/<run>/ui1|ui2|ui3.
- Nen coi day la legacy; uu tien UI moi trong ui/views/*.py khi phat trien.
Vi du nho:
```
from pedi_oku_landslide.pipeline.runners.ui1_backend import run_UI1_5_extract_boundary
```

## Steps to Complete
1) Neu sua backend, danh gia tac dong toi legacy UI (import paths, file naming).
2) Neu can dong bo tinh nang, update ca legacy va UI moi hoac ghi ro trong log.
3) Test nhanh: chay file frontend tuong ung va kiem tra co loi import.
