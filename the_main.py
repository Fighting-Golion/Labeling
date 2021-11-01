from skimage import data
import napari

if __name__=='__main__':
    image = data.coins()[50:-50, 50:-50]
    viewer = napari.view_image(image, name='coins2')
    napari.run()


