import matplotlib.pyplot as plt
import data_loader.training_data_loader as tdl

vehicles_mtx, non_vehicles_mtx = tdl.load_training_data()

a = vehicles_mtx[0, :]
print(a.shape)

plt.imshow(a)
plt.show()
