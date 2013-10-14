import sys

try:
	from OpenGL.GLUT import *
	from OpenGL.GL import *
	from OpenGL.GLU import *
except:
	print '''
ERROR: PyOpenGL not installed properly.
		'''

from abstractscene import abstractScene


class connect4Scene(abstractScene):
	def __init__(self, depth, ab, k, cols, rows):
		super(connect4Scene, self).__init__()

	def initialize(self):
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
		glLoadIdentity ()
		glFrustum (-1.0, 1.0, -1.0, 1.0, 1.5, 20.0)
		glMatrixMode (GL_MODELVIEW)
	def glutKeyPressEvent(self, key, x, y):
		print key
	def glutKeyReleaseEvent(self, key, x, y):
		if key == chr(27):
			sys.exit(0)
	def glutMousePassiveMoveEvent(self, x, y):pass
	def glutMouseActiveMoveEvent(self, x, y):pass
	def glutMouseClickEvent(self, button, state, x, y):pass


	# GENERATE BOARD VBO
	