# Enhanced Dataset Wrappers

---

* Usage

```
import data_prep_utils
data_prep_utils.set_root("/content/drive/Shareddrives/{your-data-root-folder}/")
from data_prep_utils import covid_19_radiography_dataset
data = covid_19_radiography_dataset.get_torch_dataset(transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(data, batch_size=64, shuffle=True)
```

---
