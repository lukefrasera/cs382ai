import sys
from numpy import *

try:
	from OpenGL.GLUT import *
	from OpenGL.GL import *
	from OpenGL.GLU import *
	from OpenGL.GL import shaders
	from OpenGL.arrays import vbo
	from PIL import Image
	from ctypes import *
except:
	print '''
ERROR: PyOpenGL not installed properly.
		'''

from abstractscene import abstractScene

def eval_f(state):pass



class connect4Scene(abstractScene):
	def __init__(self, depth, ab, k, cols, rows):
		super(connect4Scene, self).__init__()
		self.rows = rows
		self.cols = cols

	def initialize(self):
		# SHADERS
		VERTEX_SHADER = shaders.compileShader("""#version 110
		void main() {
			gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
		}""", GL_VERTEX_SHADER)

		FRAGMENT_SHADER = shaders.compileShader("""#version 110
		void main() {
			gl_FragColor = vec4( 0, 1, 0, 1 );
		}""", GL_FRAGMENT_SHADER)


		self.shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)
		self.genBoard()
		# self.dispBoardVbo = vbo.VBO(self.vertices)

		self.vbo = glGenBuffers(1)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
		glBufferData(GL_ARRAY_BUFFER, self.size*4, self.vertices, GL_STATIC_DRAW)
		# self.position_location = glGetAttribLocation(self.shader, 'gl_Position')
		# glEnableClientState(GL_VERTEX_ARRAY)
		# glEnableClientState(GL_TEXTURE_COORD_ARRAY)

		# glVertexPointer(3, GL_FLOAT, 4*2, self.dispBoardVbo)
		# glTexCoordPointer(2, GL_FLOAT, 4*3, self.dispBoardVbo + 3)

		# LOAD IMAGES
		
		glEnable(GL_TEXTURE_2D)
		glDisable(GL_DEPTH_TEST)
		glEnable(GL_BLEND)
		glEnable(GL_POLYGON_SMOOTH)
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
		glClearColor(0, 0, 0, 1)
		gluOrtho2D(0, self.cols, 0, self.rows)
		
		self.loadImages()

	def update(self):pass

	def render(self):

		glClearColor(0, 0, 0, 1)
		glClear(GL_COLOR_BUFFER_BIT)
		# glBindTexture(GL_TEXTURE_2D, self.texIds[0])
		glUseProgram(self.shader)
		glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

		glEnableClientState(GL_VERTEX_ARRAY)
		# glEnableClientState(GL_TEXTURE_COORD_ARRAY)

		glVertexPointer(3, GL_FLOAT, 0, None)
		# glTexCoordPointer(2, GL_FLOAT, 4*3, self.dispBoardVbo + 3)

		glDrawArrays(GL_TRIANGLES, 0, self.size)

		glDisableClientState(GL_VERTEX_ARRAY)
		# glDisableClientState(GL_TEXTURE_COORD_ARRAY)


	def resize(self, w, h):
		glViewport (0, 0, w, h)
		glMatrixMode (GL_PROJECTION)
		gluOrtho2D(0, self.cols, 0, self.rows)
		glLoadIdentity ()
		glMatrixMode (GL_MODELVIEW)

	def glutKeyPressEvent(self, key, x, y):
		pass
	def glutKeyReleaseEvent(self, key, x, y):
		if key == chr(27):
			sys.exit(0)
	def glutMousePassiveMoveEvent(self, x, y):pass
	def glutMouseActiveMoveEvent(self, x, y):pass
	def glutMouseClickEvent(self, button, state, x, y):pass
		# if self.pTurn:
		# 	print self.colClicked(x,y)
		# 	# PLAY TURN
		# 	self.playTurn(colClicked)

	def loadImages(self):

		imageBoard = Image.open('board.png')
		imageRed = Image.open('red.png')
		imageBlack = Image.open('Black.png')

		images = []
		images.append(Image.open('board.png'))
		images.append(Image.open('red.png'))
		images.append(Image.open('Black.png'))

		self.texIds = glGenTextures(3)

		for i, id_ in enumerate(self.texIds):
			x, y, image = images[i].size[0], images[i].size[1], images[i].convert("RGBA").tostring("raw", "RGBA", 0,-1)

			glBindTexture(GL_TEXTURE_2D, i)
			glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
			# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
			# glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, x, y, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
			


	# GENERATE BOARD VBO
	def genBoard(self):
		# GENERATE ARRAY OF QUADS
		# self.size = self.rows * self.cols * 6
		# self.vertices = empty((self.size, 5), dtype=float)
		# for i in range(self.rows):
		# 	for j in range(self.cols):
		# 		quad = array(self.square(j,i))
		# 		for k in  range(6):
		# 			self.vertices[(i*self.cols + j)*6 + k] = quad[k]

		self.size = 6
		self.vertices = array([[-0.5, -0.5, 0],
						[0.5, 0.5, 0],
						[-0.5, 0.5, 0],
						[-0.5, -0.5, 0],
						[0.5, 0.5, 0],
						[0.5, -0.5, 0]])

	@staticmethod
	def square(x,y):
		'''		position	     |texture coor'''
		return [[x-0.5, y-0.5, 0, 0, 0],
				[x-0.5, y+0.5, 0, 0, 1],
				[x+0.5, y+0.5, 0, 1, 1],
				[x-0.5, y-0.5, 0, 0, 0],
				[x+0.5, y+0.5, 0, 1, 1],
				[x+0.5, y-0.5, 0, 1, 0]]