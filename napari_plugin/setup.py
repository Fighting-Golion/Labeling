from setuptools import setup,find_packages
setup(name='acLearn',
      version='1.0.0',
      description='this is a active learning package',
      author='golion',
      author_email='2446049676@qq.com',
      packages=find_packages(),
entry_points={'napari.plugin': 'open = acLearn.train_and_select'}

      )