from PIL import Image

# load image
imageNames = open("zzfileNames.txt", "r")

for i in range(19992): 
        imageName = imageNames.readline()
        imageName = imageNames.readline()
        imageName = imageName.strip()

        if imageName != "zzfileNames.txt" and imageName != "zzimagePreprocess.py":
                image = Image.open(imageName)
                width, height = image.size

                if width == height:
                        continue

                # create a cropped image
                cropped = image.crop((16, 0, 112, 96))

                pathName = "D:/Assignment 4/Assignment 4/homework5_for_python_3(1)/homework5_for_python_3/drive_data/"
                pathName = pathName + imageName

                # save image
                cropped.save(pathName, format='PNG')
                image.close()
        else:
                imageNames.close()
                break

# # # show cropped image
# cropped.show()