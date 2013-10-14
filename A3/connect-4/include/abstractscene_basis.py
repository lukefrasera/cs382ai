from OpenGL.GLUT import *
from OpenGL.GL import *
from OpenGL.GLU import *
import sys


class abstractScene:
	def __init__(self):
		self.w = 500
		self.h = 500

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

	def glutKeyPress(self, key, x, y):
		if key == chr(27):
			sys.exit(0)
	def glutKeyRelease(self, key, x, y):pass
	def glutMouse(self, x, y):pass
	def glutMouseClick(self, button, state, x, y):pass

class glapplication:
	def __init__(self, scene):
		self.scene = scene

		self.initializeScene()
		self.initializeCallbacks()
		glutMainLoop()

	def initializeScene(self):
		glutInit(sys.argv)
		glutInitDisplayMode(GLUT_SINGLE | GLUT_DEPTH | GLUT_RGBA)
		glutInitWindowSize(self.scene.w, self.scene.h)
		glutCreateWindow("Connect 4")
		self.scene.initialize()

	def cleanupScene(self):pass

	def initializeCallbacks(self):
		glutDisplayFunc(self.scene.render)
		glutReshapeFunc(self.scene.resize)
		glutIdleFunc(self.scene.update)
		glutKeyboardFunc(self.scene.glutKeyPress)
		glutKeyboardUpFunc(self.scene.glutKeyRelease)
		glutMouseFunc(self.scene.glutMouseClick)
		glutPassiveMotionFunc(self.scene.glutMouse)


def main():
	scene = abstractScene()
	app = glapplication(scene)


if __name__ == '__main__':
	main()