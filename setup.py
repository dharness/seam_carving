from setuptools import setup, find_packages


long_description = """
seam_carver is a small tool for retargetting images to any dimension greater or smaller.
It uses the process of seam carving as originally described by Shai Avidan & Ariel Shamir in http://perso.crans.org/frenoy/matlab2012/seamcarving.pdf

A combination of the gradient energy (determined with the sobel filter) and a simple color energy is
used to determine the least important seam. Additon of seams occurs by the same mechanism as described
in the paper.
"""


setup(name='seam_carver',
      description='A small tool for retargetting images to any dimension greater or smaller',
      long_description=long_description,
      version='0.1.2',
      url='https://github.com/dharness/seam_carving',
      author='Dylan Harness',
      author_email='dharness.engineer@gmail.com',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Developers',
          'Programming Language :: Python :: 3'
      ],
      keywords='seam seamcarving carving liquidresizing retarget retargetting retargeting',
      packages=find_packages(),
      install_requires=[
          'scipy>=0.19.1',
          'numpy>=1.13.1',
          'Pillow>=4.2.1'
      ],
      python_requires='>=3'
)