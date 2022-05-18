import sys, re, subprocess
from termcolor import colored

from PIL import Image as PImage, ImageDraw, ImageColor
import pytesseract

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal




class Image(object):
  def __init__(self):
    self.textData = []

  def setTextData(self, data):
    self.textData = data

  def getTextData(self):
    return self.textData

  def getImage(self):
    return self.image

  def setImage(self, image):
    self.image = image

  def getOriginalImage(self):
    return self.originalImage


class SpecificationImage(Image):
  def __init__(self, specPath):
    super().__init__()
    self.specPath = specPath
    self.filePath = "tmp/tmpName."
    self.renderFile("png")
    self.originalImage = PImage.open(self.filePath + "png")
    self.image = self.originalImage.copy()

  def getImagePath(self, fileFormat):
    self.renderFile(fileFormat)
    return self.filePath + fileFormat

  def renderFile(self, fileFormat):
    subprocess.run([
      "vl2" + fileFormat, # cli tool for rendering vega-lite specs
      self.specPath, # path to the specification
      self.filePath + fileFormat, # temporary output path
      "-s 4" # scale
    ])


class OriginalImage(Image):
  def __init__(self, imagePath):
    super().__init__()
    self.imagePath = imagePath
    self.originalImage = PImage.open(self.imagePath)
    self.image = self.originalImage.copy()





class TextStep(object):
  def __init__(self, specImg, origImg):
    self.specImg = specImg
    self.origImg = origImg

  def main(self):
    self.processOriginal()
    self.processSpecification()
    self.match()

  def processOriginal(self):
    self.findOrigText()
    self.removeText(self.origImg)
    origTextData = self.origImg.getTextData()
    self.origImg.setTextData([])
    self.findOrigText(270)
    self.removeText(self.origImg)
    self.origImg.setTextData(origTextData + self.origImg.getTextData())

  def processSpecification(self):
    self.findSpecText()
    self.removeText(self.specImg)

  def match(self):
    ow = self.origImg.getImage().width
    oh = self.origImg.getImage().height
    sw = self.specImg.getImage().width
    sh = self.specImg.getImage().height

    totalWords = len(self.origImg.getTextData())
    matchedWords = 0

    specBase = self.specImg.getOriginalImage().copy()
    specOverlay = PImage.new("RGBA", specBase.size, (0, 0, 0, 100))
    specDraw = ImageDraw.Draw(specOverlay, "RGBA")

    origBase = self.origImg.getOriginalImage().copy()
    origOverlay = PImage.new("RGBA", origBase.size, (0, 0, 0, 100))
    origDraw = ImageDraw.Draw(origOverlay, "RGBA")

    colors = list(ImageColor.colormap)

    for index, origBox in enumerate(self.origImg.getTextData()):
      OxCenter = (origBox['x0'] + origBox['width']/2)/ow
      OyCenter = (origBox['y0'] + origBox['height']/2)/oh

      rgbcolor = ImageColor.getrgb(colors[index])

      origDraw.rectangle(
        [
          origBox['x0'],
          origBox['y0'],
          origBox['x0'] + origBox['width'],
          origBox['y0'] + origBox['height'],
        ],
        fill=rgbcolor + (100,)
      )

      for specBox in self.specImg.getTextData():
        x0 = specBox['x0']/sw
        y0 = specBox['y0']/sh
        x1 = (specBox['x0'] + specBox['width'])/sw
        y1 = (specBox['y0'] + specBox['height'])/sh
        # print(OxCenter, x0, x1)
        if OxCenter >= x0 and OxCenter <= x1 and OyCenter >= y0 and OyCenter <= y1:

          specDraw.rectangle(
            [
              specBox['x0'],
              specBox['y0'],
              specBox['x0'] + specBox['width'],
              specBox['y0'] + specBox['height'],
            ],
            fill=rgbcolor + (100,)
          )

          if origBox['value'] in specBox['value']:
            matchedWords += 1
          else:
            print(origBox['value'] + " != " + specBox['value'])
            specDraw.rectangle(
              [
                specBox['x0'],
                specBox['y0'],
                specBox['x0'] + specBox['width'],
                specBox['y0'] + specBox['height'],
              ],
              outline="red",
              width=2
            )
            origDraw.rectangle(
              [
                origBox['x0'],
                origBox['y0'],
                origBox['x0'] + origBox['width'],
                origBox['y0'] + origBox['height'],
              ],
              outline="red",
              width=2
            )


    specPrev = PImage.alpha_composite(specBase, specOverlay)
    origPrev = PImage.alpha_composite(origBase, origOverlay)
    specPrev.show()
    origPrev.show()

    print("{0}% match".format(matchedWords*100/totalWords))



  def findSpecText(self):
    specPath = self.specImg.getImagePath("pdf")

    textData = []

    specImg = self.specImg.getImage().copy()
    drawing = ImageDraw.Draw(specImg)

    for page_layout in extract_pages(specPath):
      for element in page_layout:
        if isinstance(element, LTTextBoxHorizontal):

          word = element.get_text().replace(" ", "")

          # check if word is horizontal
          match = re.search("^(.\n)+", word)
          if match is not None:
            word = word.replace("\n", "")[::-1]

          x0 = element.x0
          y0 = specImg.height - element.height - element.y0
          w = element.width
          h = element.height

          textData.append({ "x0": x0, "y0": y0, "height": h, "width": w, "value": word })

    self.specImg.setTextData(textData)


  def findOrigText(self, rotation=0):
    # work on a copy of the given image
    workingCopy = self.origImg.getImage().copy().rotate(rotation, expand=1)

    imageHeight = workingCopy.height
    imageWidth = workingCopy.width

    # run the OCR algorithm
    data = pytesseract.image_to_data(workingCopy, output_type=pytesseract.Output.DICT)

    textData = []

    # for each word in the OCR result
    for index, word in enumerate(data['text']):
      # words must be longer than one character
      if len(word) <= 1:
        continue

      match = re.search(".+[a-zA-Z0-9].+", word)
      # words must contain at least one alphanumeric letter
      if match is None:
        continue

      x0 = data['left'][index]
      y0 = data['top'][index]
      h = data['height'][index]
      w = data['width'][index]

      if rotation == 270:
        x0 = data['top'][index]
        y0 = imageWidth - data['left'][index] - data['width'][index]
        h = data['width'][index]
        w = data['height'][index]

      textData.append({ "x0": x0, "y0": y0, "height": h, "width": w, "value": word })

    self.origImg.setTextData(textData)


  # covers text on the given image with black boxes
  def removeText(self, image, rotation=0):
    # prepare to draw on the image
    workingCopy = image.getImage().copy()
    drawing = ImageDraw.Draw(workingCopy)

    # temporary for different colors
    colors = list(ImageColor.colormap)

    for index, box in enumerate(image.getTextData()):
      # draw a box over the identified text
      drawing.rectangle(
        [
          box['x0'],
          box['y0'],
          box['x0'] + box['width'],
          box['y0'] + box['height'],
        ],
        fill=colors[index]
      )

    workingCopy.show()

    image.setImage(workingCopy)





def main(args):
  # init both images
  specImg = SpecificationImage(args[0])
  origImg = OriginalImage(args[1])


  textStep = TextStep(specImg, origImg)
  textStep.main()

  # origImg.getImage().show()
  # specImg.getImage().show()



if __name__ == '__main__':
  if len(sys.argv) < 3:
    print(colored("Usage: cmd <specification path> <image path>", 'white'))
  else:
    args = sys.argv
    args.pop(0)
    main(args)