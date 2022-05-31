import sys, re, subprocess, json, logging, uuid
from termcolor import colored
import argparse

from skimage.metrics import structural_similarity
from PIL import Image as PImage, ImageDraw, ImageColor
import pytesseract
import cv2 as cv
import numpy as np

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal


logging.basicConfig(format='[%(levelname)s] %(message)s', level=logging.INFO)

log = logging.getLogger(__name__)

# log to file - default log is to terminal
# log.addHandler(logging.FileHandler("himmerland.log"))


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


class SpecificationImage(Image):
  def __init__(self, specPath, scale):
    super().__init__()
    self.specPath = specPath
    self.modifiedSpecPath = "tmp/modifiedSpec.json"
    self.filePath = "tmp/specImage."
    self.scale = scale

    with open(self.specPath) as spec:
      self.spec = json.load(spec)
      with open(self.modifiedSpecPath, "w") as modSpec:
        modSpec.truncate()
        json.dump(self.spec, modSpec)
        modSpec.close()
      spec.close()

    self.renderFile()
    self.image = PImage.open(self.filePath + "png")


  def getImagePath(self, fileFormat):
    self.renderFile(fileFormat)
    return self.filePath + fileFormat

  def getSpec(self):
    return self.spec

  def setSpec(self, spec):
    self.spec = spec
    with open(self.modifiedSpecPath, "w") as specFile:
      specFile.truncate()
      json.dump(spec, specFile)
      specFile.close()

  def renderFile(self, fileFormat="png"):
    subprocess.run([
      "vl2" + fileFormat, # cli tool for rendering vega-lite specs
      self.modifiedSpecPath, # path to the specification
      self.filePath + fileFormat, # temporary output path
      "-s {0}".format(self.scale) # scale
    ])
    self.image = PImage.open(self.filePath + "png")


class OriginalImage(Image):
  def __init__(self, imagePath):
    super().__init__()
    self.imagePath = imagePath
    self.image = PImage.open(self.imagePath)




class DimensionsStep(object):
  def __init__(self, specImg, origImg, scale):
    self.specImg = specImg
    self.origImg = origImg
    self.scale = scale

  def main(self):
    log.info("* Running DimensionsStep...")
    origWidth, origHeight = self.getImgDimensions(self.origImg)
    log.debug("Size of original image {0}x{1}".format(origWidth, origHeight))

    self.renderSpecImageWithDimensions(origWidth, origHeight)

    specWidth, specHeight = self.getImgDimensions(self.specImg)
    log.debug("Size of modified spec image {0}x{1}".format(specWidth, specHeight))

    widthDiff = specWidth - origWidth
    heightDiff = specHeight - origHeight

    log.debug("Image differs by {0} in width and {1} in height".format(widthDiff, heightDiff))

    newWidth = origWidth - widthDiff/self.scale
    newHeight = origHeight - heightDiff/self.scale

    self.renderSpecImageWithDimensions(newWidth, newHeight)

    specWidth, specHeight = self.getImgDimensions(self.specImg)

    widthDiff = (specWidth - origWidth)/self.scale
    heightDiff = (specHeight - origHeight)/self.scale

    newWidth = newWidth - widthDiff
    newHeight = newHeight - heightDiff

    self.renderSpecImageWithDimensions(newWidth, newHeight)

    log.info("= Specification dimensions calculated {0}x{1}".format(newWidth, newHeight))

  def getImgDimensions(self, img):
    size = img.getImage().size
    return size[0], size[1]
    # return int(size[0] / scale), int(size[1] / scale)

  def renderSpecImageWithDimensions(self, width, height):
    spec = self.specImg.getSpec()
    spec['width'] = width
    spec['height'] = height
    self.specImg.setSpec(spec)
    self.specImg.renderFile()



class ColorStep(object):
  def __init__(self, specImg, origImg):
    self.specImg = specImg
    self.origImg = origImg
    self.themes = [
      "default",
      "dark",
      "excel",
      "fivethirtyeight",
      "ggplot2",
      "googlecharts",
      "latimes",
      "quartz",
      "urbaninstitute",
      "vox"
    ]

  def getTheme(self, theme):
    with open("vega-themes/{0}.json".format(theme)) as theme:
      parsed = json.load(theme)
      theme.close()
      return parsed

  def applyTheme(self, theme):
    themeDict = self.getTheme(theme)
    spec = self.specImg.getSpec()
    spec['config'] = themeDict
    self.specImg.setSpec(spec)
    self.specImg.renderFile()

  def getHistogram(self, image):
    # https://docs.opencv.org/3.4/d8/dc8/tutorial_histogram_comparison.html
    img = cv.cvtColor(np.array(image), cv.COLOR_RGB2BGR)
    histogram = cv.calcHist([img],[0,1,2], None, [256,256,256], [0,256,0,256,0,256])
    cv.normalize(histogram, histogram, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    return histogram

  def main(self):
    log.info("* Running ColorStep...")

    originalImage = self.origImg.getImage()
    origHistogram = self.getHistogram(originalImage)

    bestScore = 0
    bestTheme = None
    for i, theme in enumerate(self.themes):
      self.applyTheme(theme)
      specificationImage = self.specImg.getImage()
      specHistogram = self.getHistogram(specificationImage)

      # Method: Correlation
      score = cv.compareHist(origHistogram, specHistogram, 0)
      # for compare_method in range(4):
      #   score = cv.compareHist(origHistogram, specHistogram, compare_method)

      log.debug("Comparing theme '{0}' with a score of {1}".format(theme, score))

      if score > bestScore:
        bestScore = score
        bestTheme = theme

    log.info("= Applying theme '{0}' with the highest score of {1}%".format(bestTheme, bestScore*100, 2))
    self.applyTheme(bestTheme)




class TextStep(object):
  def __init__(self, specImg, origImg):
    self.specImg = specImg
    self.origImg = origImg

    self.specInitialImg = specImg.getImage().copy()
    self.origInitialImg = origImg.getImage().copy()

  def main(self):
    log.info("* Running TextStep...")

    self.processOriginal()
    self.processSpecification()
    self.removeText(self.origImg, self.specImg.getTextData())
    self.removeText(self.specImg, self.origImg.getTextData())
    self.compare(False)

  def processOriginal(self):
    self.findOrigText()
    self.removeText(self.origImg, self.origImg.getTextData())
    origTextData = self.origImg.getTextData()
    self.origImg.setTextData([])
    self.findOrigText(270)
    self.removeText(self.origImg, self.origImg.getTextData())
    self.origImg.setTextData(origTextData + self.origImg.getTextData())

    log.debug("Found {0} textual elements in original image".format(len(self.origImg.getTextData())))

  def processSpecification(self):
    self.findSpecText()
    self.removeText(self.specImg, self.specImg.getTextData())

    log.debug("Found {0} textual elements in spec pdf".format(len(self.specImg.getTextData())))

  def compare(self, showPreview=False):
    ow = self.origImg.getImage().width
    oh = self.origImg.getImage().height
    sw = self.specImg.getImage().width
    sh = self.specImg.getImage().height

    totalWords = len(self.origImg.getTextData())
    matchedWords = 0

    specBase = self.specInitialImg
    specOverlay = PImage.new("RGBA", specBase.size, (0, 0, 0, 100))
    specDraw = ImageDraw.Draw(specOverlay, "RGBA")

    origBase = self.origInitialImg
    origOverlay = PImage.new("RGBA", origBase.size, (0, 0, 0, 100))
    origDraw = ImageDraw.Draw(origOverlay, "RGBA")

    colors = list(ImageColor.colormap)

    for index, origBox in enumerate(self.origImg.getTextData()):
      origText = origBox['value'].lower().strip()

      OxCenter = (origBox['x0'] + origBox['width']/2)/ow
      OyCenter = (origBox['y0'] + origBox['height']/2)/oh

      rgbcolor = ImageColor.getrgb(colors[index])

      fillColor = rgbcolor + (100,)

      if showPreview:
        origDraw.rectangle(
          [
            origBox['x0'],
            origBox['y0'],
            origBox['x0'] + origBox['width'],
            origBox['y0'] + origBox['height'],
          ],
          fill=fillColor
        )

      for specBox in self.specImg.getTextData():
        specText = specBox['value'].lower()

        x0 = specBox['x0']/sw
        y0 = specBox['y0']/sh
        x1 = (specBox['x0'] + specBox['width'])/sw
        y1 = (specBox['y0'] + specBox['height'])/sh
        if OxCenter >= x0 and OxCenter <= x1 and OyCenter >= y0 and OyCenter <= y1:

          if showPreview:
            specDraw.rectangle(
              [
                specBox['x0'],
                specBox['y0'],
                specBox['x0'] + specBox['width'],
                specBox['y0'] + specBox['height'],
              ],
              fill=fillColor
            )

          if origText in specText:
            matchedWords += 1
          else:
            log.debug("Words '{0}' and '{1}' were at the same location but did not match".format(origText.replace("\n","\\n"), specText.replace("\n","\\n")))
            if showPreview:
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

    if showPreview:
      specPrev = PImage.alpha_composite(specBase, specOverlay)
      origPrev = PImage.alpha_composite(origBase, origOverlay)
      specPrev.show()
      origPrev.show()

    log.info("= Matched textual elements with a score of {0}%".format(matchedWords*100/totalWords))



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
  def removeText(self, image, textData, rotation=0):
    # prepare to draw on the image
    workingCopy = image.getImage().copy()
    drawing = ImageDraw.Draw(workingCopy)

    # temporary for different colors
    # colors = list(ImageColor.colormap)

    # fillColor = colors[index]
    fillColor = (0,0,0)

    for index, box in enumerate(textData):
      # draw a box over the identified text
      drawing.rectangle(
        [
          box['x0'],
          box['y0'],
          box['x0'] + box['width'],
          box['y0'] + box['height'],
        ],
        fill=fillColor
      )

    # workingCopy.show()

    image.setImage(workingCopy)


class FinalStep(object):
  def __init__(self, specImg, origImg):
    self.specImg = specImg
    self.origImg = origImg

  def main(self):
    log.info("* Running FinalStep...")

    specImg = self.specImg.getImage().copy().convert('RGBA')
    origImg = self.origImg.getImage().copy().convert('RGBA')

    overlay = PImage.new('RGBA',specImg.size,(255,255,255,150))

    specImg.alpha_composite(overlay)
    origImg.alpha_composite(overlay)

    specImg = specImg.convert('RGB')
    origImg = origImg.convert('RGB')

    specImg = np.array(specImg)
    origImg = np.array(origImg)

    difference = cv.subtract(specImg, origImg)

    # color the mask red
    Conv_hsv_Gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    ret, mask = cv.threshold(Conv_hsv_Gray, 0, 255,cv.THRESH_BINARY_INV |cv.THRESH_OTSU)

    fillColor = [255, 0, 0]

    difference[mask != 255] = fillColor
    specImg[mask != 255] = fillColor
    origImg[mask != 255] = fillColor

    PImage.fromarray(difference).show()
    PImage.fromarray(specImg).show()
    PImage.fromarray(origImg).show()


def program(specPath, visPath, scale):
  # init both images
  specImg = SpecificationImage(specPath, scale)
  origImg = OriginalImage(visPath)

  colorStep = ColorStep(specImg, origImg)
  colorStep.main()

  dimensionsStep = DimensionsStep(specImg, origImg, scale)
  dimensionsStep.main()

  textStep = TextStep(specImg, origImg)
  textStep.main()

  finalStep = FinalStep(specImg, origImg)
  finalStep.main()

  # origImg.getImage().show()
  # specImg.getImage().show()



if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Validate a data visualization against a Vega-Lite specification.')
  parser.add_argument('-v', '--verbose', action='store_true', help='Show verbose output')
  parser.add_argument('-s', '--scale', metavar='number', nargs=None, type=int, help='Scale of original image', default=1)
  parser.add_argument('-c', '--spec', metavar='file', nargs=1, type=argparse.FileType('r'), help='Vega-Lite specification', required=True)
  parser.add_argument('-i', '--vis', metavar='file', nargs=1, type=argparse.FileType('r'), help='Data visualization to be validated', required=True)

  args = parser.parse_args()

  specPath = args.spec[0].name
  visPath = args.vis[0].name
  scale = args.scale
  verbose = args.verbose

  if verbose:
    log.setLevel(logging.DEBUG)

  program(specPath, visPath, scale)
