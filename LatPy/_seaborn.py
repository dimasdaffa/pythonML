import seaborn as sns
import matplotlib.pyplot as plt

# Memuat dataset 'tips' dari Seaborn
tips = sns.load_dataset('tips')

# Membuat histogram dengan KDE
sns.histplot(tips['total_bill'], kde=True)
plt.title('Histogram Total Bill')
plt.xlabel('Total Bill')
plt.ylabel('Frequency')
plt.show()
