from PIL import Image, ImageDraw
import math


MAX_ITER = 80

# Image size (pixels)
WIDTH = 600
HEIGHT = 400

# Plot window
RE_START = -2
RE_END = 1
IM_START = -1
IM_END = 1

palette = []


def mandelbrot(c):
    z = 0
    n = 0
    while abs(z) <= 2 and n < MAX_ITER:
        z = z*z - c
        n += 1
    return n

if __name__ == '__main__':

    im = Image.new('RGB', (WIDTH, HEIGHT), (0, 0, 0))
    draw = ImageDraw.Draw(im)

    for x in range(0, WIDTH):
        for y in range(0, HEIGHT):
            # Convert pixel coordinate to complex number
            c = complex(math.exp(RE_START + (x / WIDTH) * (RE_END - RE_START)),
                        math.exp(IM_START + (y / HEIGHT) * (IM_END - IM_START)))
            # Compute the number of iterations
            m = mandelbrot(c)
            # The color depends on the number of iterations
            color = 255 - int(m * 255 / MAX_ITER)
            # Plot the point
            draw.point([x, y], (color, color, color))

    im.show()
    #('output.png', 'PNG')