# https://izdebski.edu.pl/kategorie/Informatyka/Cwiczenie_02.pdf
# metoda obliczenia punktów przecięcia j.w.
from math import sqrt
import time
import numpy as np
import matplotlib.pyplot as plt

# Configurations
Amount_emmiters = 20  # Ilość żródeł
Amount_pixels = 20  # ilość pixeli w rzędzie
ITERATIONS = 300  # ilość iteracji w algorytmie Kaczmarza
WSP_LAMBDA = np.logspace(0.7, 0.0001, num=ITERATIONS)
LowBoundX = LowBoundY = -1  # dolne granice x, y
HiBoundX = HiBoundY = 1  # górne granice x, y
MULTIPLIER = 10e+7 # mnożnik

plt.plot(WSP_LAMBDA)
plt.show()
# Global functions
def NUM(num):
    return int(round(num))


def FNUM(num):
    return int(round(num * MULTIPLIER))


def FLNUM(lst):
    return [FNUM(x) for x in lst]


# Przykład 1
# wiechrzołki prostokąta A
p1_wiech_prost_A = {'A': [-0.7, -0.5], 'B': [-0.4, -0.5], 'C': [-0.7, 0.2], 'D': [-0.4, 0.2]}
p1_waga_post_A = 1
# wiechrzołki prostokąta B
p1_wiech_prost_B = {'A': [-0.2, -0.1], 'B': [0.2, -0.1], 'C': [-0.2, 0.1], 'D': [0.2, 0.1]}
p1_waga_post_B = 2
# wiechrzołki prostokąta C
p1_wiech_prost_C = {'A': [-0.2, 0.3], 'B': [0.2, 0.3], 'C': [-0.2, 0.5], 'D': [0.2, 0.5]}
p1_waga_post_C = 3
# wiechrzołki prostokąta D
p1_wiech_prost_D = {'A': [0.4, 0.4], 'B': [0.7, 0.4], 'C': [0.4, 0.7], 'D': [0.7, 0.7]}
p1_waga_post_D = 4
# Emitery
p1_emitery = [[-1, -1], [-0.5, -1], [0, -1], [0.5, -1], [1, -1]]
# Dekodery
p1_detektory = [[-1, 1], [-0.5, 1], [0, 1], [0.5, 1], [1, 1]]

# Przykład 2
# wiechrzołki prostokąta A
p2_wiech_prost_A = {'A': [-0.4, -0.5], 'B': [-0.2, -0.5], 'C': [-0.4, 0.5], 'D': [-0.2, 0.5]}
p2_waga_post_A = 1
# wiechrzołki prostokąta B
p2_wiech_prost_B = {'A': [-0.2, 0.3], 'B': [0.2, 0.3], 'C': [-0.2, 0.5], 'D': [0.2, 0.5]}
p2_waga_post_B = 2
# wiechrzołki prostokąta C
p2_wiech_prost_C = {'A': [-0.2, -0.1], 'B': [0.2, -0.1], 'C': [-0.2, 0.1], 'D': [0.2, 0.1]}
p2_waga_post_C = 3
# wiechrzołki prostokąta D
p2_wiech_prost_D = {'A': [0.0, 0.1], 'B': [0.2, 0.1], 'C': [0.0, 0.3], 'D': [0.2, 0.3]}
p2_waga_post_D = 4
# wiechrzołki prostokąta E
p2_wiech_prost_E = {'A': [0.6, -0.8], 'B': [0.8, -0.8], 'C': [0.6, -0.6], 'D': [0.8, -0.6]}
p2_waga_post_E = 5


class EmitterDetector:
    def __init__(self):
        self.emitters = []
        self.detectors = []
        self.create()

    def create(self):
        step = (abs(FNUM(HiBoundX) - FNUM(LowBoundX))) / (Amount_emmiters - 1)
        x = FNUM(LowBoundX)
        self.emitters.append([x, FNUM(LowBoundY)])
        self.detectors.append([x, FNUM(HiBoundY)])
        for i in range(0, Amount_emmiters - 1, 1):
            x = x + step
            self.emitters.append([NUM(x), FNUM(LowBoundY)])
            self.detectors.append([NUM(x), FNUM(HiBoundY)])


class Beam:
    def __init__(self, emitter, detector):
        self.emitter = emitter
        self.detector = detector
        self.trajectory = None
        self.set_trajectory()

    def set_trajectory(self):
        if self.emitter[0] < self.detector[0]:
            self.trajectory = 1
        elif self.emitter[0] > self.detector[0]:
            self.trajectory = -1
        else:
            self.trajectory = 0

    def __str__(self) -> str:
        return f"emitter {self.emitter}, detector {self.detector}, trajectory {self.trajectory}"


class Rectangle:
    def __init__(self):
        self.sides = {}
        self.weight = None

    def create(self, panicles, weight):
        self.sides = {'AC': [FLNUM(panicles['A']), FLNUM(panicles['C'])],
                      'BD': [FLNUM(panicles['B']), FLNUM(panicles['D'])],
                      'AB': [FLNUM(panicles['A']), FLNUM(panicles['B'])],
                      'CD': [FLNUM(panicles['C']), FLNUM(panicles['D'])]}
        self.weight = weight

    def __str__(self) -> str:
        return f"sides {self.sides}, weight {self.weight}"


class Pixel:
    def __init__(self, sides, number):
        self.sides = sides
        self.number = number

    def __str__(self):
        return f"nr: {self.number} sides : {self.sides}"


class TomographStage1:
    def __init__(self):
        self.beams = []
        self.losts = []
        self.rectangles = []

    def create_rectangles_example1(self):
        rectangle1 = Rectangle()
        rectangle2 = Rectangle()
        rectangle3 = Rectangle()
        rectangle4 = Rectangle()
        rectangle1.create(p1_wiech_prost_A, p1_waga_post_A)
        rectangle2.create(p1_wiech_prost_B, p1_waga_post_B)
        rectangle3.create(p1_wiech_prost_C, p1_waga_post_C)
        rectangle4.create(p1_wiech_prost_D, p1_waga_post_D)
        self.rectangles.append(rectangle1)
        self.rectangles.append(rectangle2)
        self.rectangles.append(rectangle3)
        self.rectangles.append(rectangle4)

    def create_rectangles_example2(self):
        rectangle1 = Rectangle()
        rectangle2 = Rectangle()
        rectangle3 = Rectangle()
        rectangle4 = Rectangle()
        rectangle5 = Rectangle()
        rectangle1.create(p2_wiech_prost_A, p2_waga_post_A)
        rectangle2.create(p2_wiech_prost_B, p2_waga_post_B)
        rectangle3.create(p2_wiech_prost_C, p2_waga_post_C)
        rectangle4.create(p2_wiech_prost_D, p2_waga_post_D)
        rectangle5.create(p2_wiech_prost_E, p2_waga_post_E)
        self.rectangles.append(rectangle1)
        self.rectangles.append(rectangle2)
        self.rectangles.append(rectangle3)
        self.rectangles.append(rectangle4)
        self.rectangles.append(rectangle5)

    def create_beams(self) -> None:
        emitter_detector = EmitterDetector()
        for emitter in emitter_detector.emitters:
            for detector in emitter_detector.detectors:
                beam = Beam(emitter=emitter, detector=detector)
                self.beams.append(beam)

    def bounding_rectangles(self, beam, side) -> bool:
        pkt_A, pkt_B = beam.emitter, beam.detector
        pkt_C, pkt_D = side

        p1Xmin = min(pkt_A[0], pkt_B[0])
        p1Xmax = max(pkt_A[0], pkt_B[0])
        p1Ymin = min(pkt_A[1], pkt_B[1])
        p1Ymax = max(pkt_A[1], pkt_B[1])

        p2Xmin = min(pkt_C[0], pkt_D[0])
        p2Xmax = max(pkt_C[0], pkt_D[0])
        p2Ymin = min(pkt_C[1], pkt_D[1])
        p2Ymax = max(pkt_C[1], pkt_D[1])

        if p1Xmax < p2Xmin or p1Xmin > p2Xmax or p1Ymax < p2Ymin or p1Ymin > p2Ymax:
            return True
        else:
            return False

    def parameters_t1t2(self, beam, side):
        pkt_A, pkt_B = beam.emitter, beam.detector
        pkt_C, pkt_D = side
        Xac = pkt_C[0] - pkt_A[0]
        Xcd = pkt_D[0] - pkt_C[0]
        Xab = pkt_B[0] - pkt_A[0]
        Yab = pkt_B[1] - pkt_A[1]
        Ycd = pkt_D[1] - pkt_C[1]
        Yac = pkt_C[1] - pkt_A[1]
        try:
            t1 = ((Xac * Ycd) - (Yac * Xcd)) / ((Xab * Ycd) - (Yab * Xcd))
        except:
            return False
        try:
            t2 = ((Xac * Yab) - (Yac * Xab)) / ((Xab * Ycd) - (Yab * Xcd))
        except:
            return False
        return t1, t2

    def intersection_point(self, t1, t2, beam):  # zwraca punkt przecięcia dwóch odcinków
        pkt_A, pkt_B = beam.emitter, beam.detector
        Xab = pkt_B[0] - pkt_A[0]
        Yab = pkt_B[1] - pkt_A[1]

        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            x = NUM(pkt_A[0] + (t1 * Xab))
            y = NUM(pkt_A[1] + (t1 * Yab))
            return x, y
        return False

    def vector_length(self, inter_points):  # oblicza długość wektora
        if len(inter_points) >= 3:
            inter_points = list(set(inter_points))
        if len(inter_points) == 1:
            return 0
        a, b = inter_points
        x0, y0 = a
        x1, y1 = b
        return NUM(sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

    def create_losts(self):
        time0 = time.time()
        for beam in self.beams:
            losts = 0
            for rectangle in self.rectangles:
                inter_points = self.find_common_dots(beam, rectangle.sides)
                if len(inter_points) > 0:
                    section_length = self.vector_length(inter_points)
                    losts = losts + (section_length * rectangle.weight)
            self.losts.append(NUM(losts))
        time1 = time.time()
        print('Total time Stage 1:', time1 - time0)

    def find_common_dots(self, beam, sides):
        inter_points = []
        for key, side in enumerate(sides):
            if not self.bounding_rectangles(beam, sides[side]):
                try:
                    t1, t2 = self.parameters_t1t2(beam, sides[side])
                    x, y = 0, 0
                    x, y = self.intersection_point(t1, t2, beam)
                    inter_points.append((x, y))
                except:
                    pass
        return inter_points

    def __str__(self):
        return f"{self.losts}"


class TomographStage2(TomographStage1):
    def __init__(self):
        super().__init__()
        self.pixels = []
        self.matrix = []
        self.create_pixels(LowBoundX, LowBoundY, HiBoundX, HiBoundY, Amount_pixels)
        print(f"ilość pixeli: {len(self.pixels)}")

    def create_pixels(self, lowBoundX, lowBoundY, HiBoundX, HiBoundY, amount):
        step = (abs(HiBoundX - lowBoundX)) / amount
        lowX = lowBoundX
        num = 0
        for _ in range(0, amount, 1):
            lowBoundX = lowX
            for _ in range(0, amount, 1):
                wiech_prost = {'A': [lowBoundX, lowBoundY], 'B': [lowBoundX + step, lowBoundY],
                               'C': [lowBoundX, lowBoundY + step], 'D': [lowBoundX + step, lowBoundY + step]}
                rectangle = Rectangle()
                rectangle.create(wiech_prost, 0)
                pixel = Pixel(rectangle.sides, num)
                self.pixels.append(pixel)
                lowBoundX = lowBoundX + step
                num += 1
            lowBoundY = lowBoundY + step

    def create_matrix(self):
        time0 = time.time()
        for beam in self.beams:
            lst_beams = []
            self.pixels_to_check = []
            pixel = self.find_first_pixel(beam)
            self.pixels_to_check.append(pixel)

            while pixel != False:
                sides_name = self.find_crossed_sides(beam, pixel)
                pixel = self.find_next_pixel(sides_name, pixel, beam)

            for num, pixel in enumerate(self.pixels_to_check):
                inter_points = self.find_common_dots(beam, pixel.sides)

                if len(inter_points) > 0:
                    section_length = self.vector_length(inter_points)
                    if section_length > 0:
                        lst_beams.append((section_length, pixel.number))
            if len(lst_beams) > 0:
                self.matrix.append(lst_beams)

        time1 = time.time()
        print('Total time Stage 2: ', time1 - time0)
        print('matrix size: ', len(self.matrix))

    def __str__(self):
        return f"{self.matrix}"

    def find_first_pixel(self, beam):
        for pixel in self.pixels:
            lst = pixel.sides['AB']
            range = list([lst[0][0], lst[1][0]])
            first_beam = list(beam.emitter)
            if range[0] <= first_beam[0] <= range[1]:
                return pixel

    def find_next_pixel(self, sides_name, pixel, beam):
        trajectory = beam.trajectory

        if trajectory == 0:
            next_nr_pixel = pixel.number + Amount_pixels
            return self.add_next_pixel(next_nr_pixel)

        if trajectory == 1 and 'CD' in sides_name and 'BD' in sides_name:
            next_nr_pixel = pixel.number + Amount_pixels + 1
            return self.add_next_pixel(next_nr_pixel)

        if trajectory == 1 and 'BD' in sides_name:
            next_nr_pixel = pixel.number + 1
            return self.add_next_pixel(next_nr_pixel)

        if trajectory == 1 and 'CD' in sides_name:
            next_nr_pixel = pixel.number + Amount_pixels
            return self.add_next_pixel(next_nr_pixel)

        if trajectory == -1 and 'AC' in sides_name and 'CD' in sides_name:
            next_nr_pixel = pixel.number + Amount_pixels - 1
            return self.add_next_pixel(next_nr_pixel)

        if trajectory == -1 and 'AC' in sides_name:
            next_nr_pixel = pixel.number - 1
            return self.add_next_pixel(next_nr_pixel)

        if trajectory == -1 and 'CD' in sides_name:
            next_nr_pixel = pixel.number + Amount_pixels
            return self.add_next_pixel(next_nr_pixel)
        # print("false1")
        return False

    def add_next_pixel(self, next_nr_pixel):
        try:
            pixel = self.pixels[next_nr_pixel]
            self.pixels_to_check.append(pixel)
            return pixel
        except:
            return False

    def find_crossed_sides(self, beam, pixel) -> list:
        sides_name = []
        inter_points = self.find_common_dots(beam, pixel.sides)

        for _, side in enumerate(pixel.sides):
            lst_side = pixel.sides[side]

            for point in inter_points:
                if lst_side[0][0] <= point[0] <= lst_side[1][0]:
                    if lst_side[0][1] <= point[1] <= lst_side[1][1]:
                        sides_name.append(side)
        return list(set(sides_name))


class TomographStage3():
    def __init__(self, matrix, losts):
        self.matrix = matrix
        self.losts = losts
        self.t_select_vector = 0
        self.t_set_zero = 0
        self.t_vector_length = 0
        # print("\n losts ", straty_energii)

    def vector_length(self, x):
        time0 = time.time()
        result = np.sqrt(x.dot(x))
        # result = np.linalg.norm(x)
        time1 = time.time()
        self.t_vector_length += time1 - time0
        return result

    def select_vector(self, ai):
        time0 = time.time()
        x = np.zeros(pow(Amount_pixels, 2), dtype=float)
        for (var, num) in ai:
            x[num] = var / MULTIPLIER
        time1 = time.time()
        self.t_select_vector += time1 - time0
        return x

    def set_zero_negative_values(self, x):
        time0 = time.time()
        x[x<0] = 0
        #x = np.where(x < 0, 0, x)
        time1 = time.time()
        self.t_set_zero += time1 - time0
        return x

    def create_result(self):
        time0 = time.time()
        matrix_size = len(self.matrix)
        x = np.zeros(pow(Amount_pixels, 2), dtype=float) # wektor początkowy
        for k in range(ITERATIONS):
            for i in range(matrix_size):
                pi = self.losts[i] / MULTIPLIER
                ai = self.select_vector(self.matrix[i])
                xa = np.dot(x.T, ai)
                sub = np.asarray(WSP_LAMBDA[k] * ((pi - xa) / (pow(self.vector_length(ai), 2))))
                sub1 = np.multiply(sub, ai)
                x = np.add(x, sub1)
                x = self.set_zero_negative_values(x) # ustawianie wartości ujemnych na zero.
        time1 = time.time()
        print('Total time Stage 3:', time1 - time0)
        return x


if __name__ == "__main__":
    """
    print('start example 1')
    Amount_emmiters = 5
    tomograph_example1 = TomographStage1()
    tomograph_example1.create_beams()
    tomograph_example1.create_rectangles_example1()
    tomograph_example1.create_losts()
    print(tomograph_example1.losts)
    print('start example 2')
    Amount_emmiters = 10
    tomograph_example2 = TomographStage1()
    tomograph_example2.create_beams()
    tomograph_example2.create_rectangles_example2()
    tomograph_example2.create_losts()
    print(tomograph_example2.losts)
    """
    print("Stage2 - create pixeles")
    tomograph2 = TomographStage2()
    tomograph2.create_beams()
    tomograph2.create_rectangles_example2()
    tomograph2.create_losts()
    tomograph2.create_matrix()
    # print(tomograph2)

    tomograph3 = TomographStage3(tomograph2.matrix, tomograph2.losts)
    pixele = tomograph3.create_result()

    print(f"matrix time: {tomograph3.t_select_vector}")
    print(f"calc vector time: {tomograph3.t_vector_length}")
    print(f"set zero time: {tomograph3.t_set_zero}")

    matrix = np.asmatrix(pixele)
    matrix.shape = (Amount_pixels, Amount_pixels)
    matrix = np.flip(matrix, 0)
    print(matrix.shape)

    plt.imshow(matrix, cmap='viridis')
    plt.show()
