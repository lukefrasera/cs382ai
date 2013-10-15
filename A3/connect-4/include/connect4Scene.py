import sys
from numpy import *

try:
	from OpenGL.GLUT import *
	from OpenGL.GL import *
	from OpenGL.GLU import *
	from OpenGL.GL import shaders
	from OpenGL.arrays import vbo
	from PIL import Image
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
		VERTEX_SHADER = shaders.compileShader("""#version 130
		void main() {
			gl_Position = gl_ModelViewProjectionMatrix * gl_Vertex;
		}""", GL_VERTEX_SHADER)

		FRAGMENT_SHADER = shaders.compileShader("""#version 130
		void main() {
			gl_FragColor = vec4( 0, 1, 0, 1 );
		}""", GL_FRAGMENT_SHADER)


		self.shader = shaders.compileProgram(VERTEX_SHADER,FRAGMENT_SHADER)
		self.genBoard()
		self.dispBoardVbo = vbo.VBO(self.vertices)
		# self.position_location = glGetAttribLocation(self.shader, 'gl_Position')

		# LOAD IMAGES
		self.loadImages()
		
		glClearColor(0.0,0.0,0.0,0.0)
		glShadeModel (GL_FLAT)
	def update(self):pass
	def render(self):
		glClear(GL_COLOR_BUFFER_BIT)
		glLoadIdentity()

		gluLookAt (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0)
		glScalef (1.0, 2.0, 1.0)
		glutWireCube (1.0)
		glFlush()
	def resize(self, w, h):
		glViewport (0, 0, w, h)
		glMatrixMode (GL_PROJECTION)
		gluOrtho2D(0, self.cols, 0, self.rows)
		glLoadIdentity ()
		glMatrixMode (GL_MODELVIEW)
	def glutKeyPressEvent(self, key, x, y):
		print key
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
		self.texIds = glGenTextures(3)

		for i in self.texIds:
			glBindTextures(GL_TEXTURE_2D, i)
			


	# GENERATE BOARD VBO
	def genBoard(self):
		# GENERATE ARRAY OF QUADS
		self.size = self.rows * self.cols * 6
		self.vertices = empty((self.size, 5), dtype=float)
		for i in range(self.rows):
			for j in range(self.cols):
				quad = array(self.square(j,i))
				for k in  range(6):
					self.vertices[(i*self.cols + j)*6 + k] = quad[k]
					print quad[k]

	@staticmethod
	def square(x,y):
		'''		position	     |texture coor'''
		return [[x-0.5, y-0.5, 0, 0, 0],
				[x-0.5, y+0.5, 0, 0, 1],
				[x+0.5, y+0.5, 0, 1, 1],
				[x-0.5, y-0.5, 0, 0, 0],
				[x+0.5, y+0.5, 0, 1, 1],
				[x+0.5, y-0.5, 0, 1, 0]]