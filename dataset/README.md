
step1 implement your datasets inherit from BaseDataset

- implement all abstract methods
- use `obtain_**` to get key attributes

step2 register your dataset via the `registery` dict within `__init__`
