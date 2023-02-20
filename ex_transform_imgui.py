# coding=utf-8
"""
Simple example using ImGui with GLFW and OpenGL
More info at:
https://pypi.org/project/imgui/
Installation:
pip install imgui[glfw]
Another example:
https://github.com/swistakm/pyimgui/blob/master/doc/examples/integrations_glfw3.py#L2
"""
import easygui
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import random
import imgui
from imgui.integrations.glfw import GlfwRenderer
import os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grafica.gpu_shape import GPUShape
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
from tooff import obj2off

__author__ = "Valeria Nahuelpan"
__license__ = "MIT"


# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True



# we will use the global controller as communication with the callback function
controller = Controller()


def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return

    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    else:
        print('Unknown key')


### 3D
global model
model = 'jarrito.off' # modelo por defecto

def readOFF(filename, color):
    vertices = []
    normals= []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line=="OFF"

        line = file.readline().strip()
        aux = line.split(' ')

        numVertices = int(aux[0])
        numFaces = int(aux[1])

        for i in range(numVertices):
            aux = file.readline().strip().split(' ')
            vertices += [float(coord) for coord in aux[0:]]
        
        vertices = np.asarray(vertices)
        vertices = np.reshape(vertices, (numVertices, 3))
        print(f'Vertices shape: {vertices.shape}')

        normals = np.zeros((numVertices,3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]
            
            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1], vertices[aux[3]][2] - vertices[aux[2]][2]]

            res = np.cross(vecA, vecB)
            normals[aux[1]][0] += res[0]  
            normals[aux[1]][1] += res[1]  
            normals[aux[1]][2] += res[2]  

            normals[aux[2]][0] += res[0]  
            normals[aux[2]][1] += res[1]  
            normals[aux[2]][2] += res[2]  

            normals[aux[3]][0] += res[0]  
            normals[aux[3]][1] += res[1]  
            normals[aux[3]][2] += res[2]  
        #print(faces)
        norms = np.linalg.norm(normals,axis=1)
        normals = normals/norms[:,None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        print(vertexData.shape)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2],:]
            vertexDataF += vertex.tolist()

            indices += [index, index + 1, index + 2]
            index += 3



        return bs.Shape(vertexDataF, indices)

def createOFFShape(pipeline, r,g, b):
    shape = readOFF(getAssetPath(model), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape

color = [0.5686,0.5451,0.4941]

def createSystem(pipeline):
    sunShape = createOFFShape(pipeline, color[0],color[1],color[2])

    sunNode = sg.SceneGraphNode("sunNode")
    sunNode.transform = tr.uniformScale(0.03)
    sunNode.childs += [sunShape]

    sunRotation = sg.SceneGraphNode("sunRotation")
    sunRotation.childs += [sunNode]

    systemNode = sg.SceneGraphNode("solarSystem")
    systemNode.childs += [sunRotation]


    return systemNode

## end 3D functions


def transformGuiOverlay(locationX, locationY, locationZ, locationRX, locationRY, locationRZ, angle, scale, color):
    global model
    global mvpPipeline
    global pipeline
    global solarSystem
    global perfMonitor
    # start new frame context
    imgui.new_frame()

    # open new window context
    imgui.begin("3D Transformations control", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE)

    # draw text label inside of current window
    imgui.text("Translate")
    edited, locationX = imgui.slider_float("Move on X", locationX, -5, 5)
    edited, locationY = imgui.slider_float("Move on Y", locationY, -5, 5)
    edited, locationZ = imgui.slider_float("Move on Z", locationZ, -5, 5)
    # edited, angle = imgui.slider_float("Angle", angle, -np.pi, np.pi)
    imgui.text("Rotate")
    edited, locationRX = imgui.slider_float("Rotate on X", locationRX, -np.pi, np.pi)
    edited, locationRY = imgui.slider_float("Rotate on Y", locationRY, -5, 5)
    edited, locationRZ = imgui.slider_float("Rotate on Z", locationRZ, -5, 5)
    imgui.text("Scale")
    edited, scale = imgui.slider_float("", scale, 0, 5)
    imgui.text("Color")
    edited, color = imgui.color_edit3("Choose new color", color[0], color[1], color[2])
    if imgui.button("Aply color"):
        solarSystem = createSystem(pipeline)
    imgui.same_line()
    # if imgui.button("White Modulation Color"):
    #     color = (1.0, 1.0, 1.0)
    #     solarSystem = createSystem(pipeline)
    global controller
    edited, checked = imgui.checkbox("wireframe", not controller.fillPolygon)
    if edited:
        controller.fillPolygon = not checked
    imgui.text("Anotations")
    imgui.button("Add simetry")
    imgui.same_line()
    imgui.button("Save simetry")
    #import nuevo modelo
    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):
            clicked_quit, selected_quit = imgui.menu_item("import new model", 'Ctrl+I', False, True)
            if clicked_quit:
                modelobj = easygui.fileopenbox() #.obj
                obj2off(modelobj, getAssetPath('newOff.off') )   #convert .obj to .off
                model = getAssetPath('newOff.off') #.off
                # Renderizamos el nuevo modelo
                solarSystem = createSystem(pipeline)
                #prueba de que se recibe el modelo
                # text = "Message to be displayed on the window GfG"
                # title = "Window Title GfG"
                # easygui.buttonbox(text, title, image = model)
                #fin prueba
            # if imgui.begin_menu('Open Recent', True):
            #     imgui.menu_item('doc.txt', None, False, True)
            #     imgui.end_menu()
            imgui.end_menu()
        imgui.end_main_menu_bar()
    
    # close current window context
    imgui.end()

    # pass all drawing comands to the rendering pipeline
    # and close frame context
    imgui.render()
    imgui.end_frame()

    return locationX, locationY, locationZ, locationRX, locationRY, locationRZ, angle, scale, color


# class ModulationTransformShaderProgram:

#     def __init__(self):

#         vertex_shader = """
#             #version 330
#             uniform mat4 transform;
#             in vec3 position;
#             in vec3 color;
#             out vec3 newColor;
#             void main()
#             {
#                 gl_Position = transform * vec4(position, 1.0f);
#                 newColor = color;
#             }
#             """

#         fragment_shader = """
#             #version 330
#             in vec3 newColor;
#             out vec4 outColor;
#             uniform vec3 modulationColor;
#             void main()
#             {
#                 outColor = vec4(modulationColor, 1.0f) * vec4(newColor, 1.0f);
#             }
#             """

#         # Binding artificial vertex array object for validation
#         VAO = glGenVertexArrays(1)
#         glBindVertexArray(VAO)

#         self.shaderProgram = OpenGL.GL.shaders.compileProgram(
#             OpenGL.GL.shaders.compileShader(vertex_shader, OpenGL.GL.GL_VERTEX_SHADER),
#             OpenGL.GL.shaders.compileShader(fragment_shader, OpenGL.GL.GL_FRAGMENT_SHADER))


#     def setupVAO(self, gpuShape):
#         glBindVertexArray(gpuShape.vao)

#         glBindBuffer(GL_ARRAY_BUFFER, gpuShape.vbo)
#         glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gpuShape.ebo)

#         # 3d vertices + rgb color specification => 3*4 + 3*4 = 24 bytes
#         position = glGetAttribLocation(self.shaderProgram, "position")
#         glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
#         glEnableVertexAttribArray(position)

#         color = glGetAttribLocation(self.shaderProgram, "color")
#         glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
#         glEnableVertexAttribArray(color)

#         # Unbinding current vao
#         glBindVertexArray(0)


#     def drawCall(self, shape, mode=GL_TRIANGLES):
#         assert isinstance(shape, GPUShape)

#         # Binding the VAO and executing the draw call
#         glBindVertexArray(shape.vao)
#         glDrawElements(mode, shape.size, GL_UNSIGNED_INT, None)

#         # Unbind the current VAO
#         glBindVertexArray(0)


if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)

    width = 1000
    height = 1000
    title = "GLFW OpenGL ImGui"
    window = glfw.create_window(width, height,title , None, None)

    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)
    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline = ls.SimpleFlatShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    mvpPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)
    print(model)
    solarSystem = createSystem(pipeline)
    # Using the same view and projection matrices in the whole application
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    viewPos = np.array([5,5,5])
    view = tr.lookAt(
            viewPos,
            np.array([0,0,0]),
            np.array([0,1,0])
        )

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), 3, 3, 3)
    
    glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.01)

    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)
    # # Creating our shader program and telling OpenGL to use it
    # pipeline = ModulationTransformShaderProgram()
    # glUseProgram(pipeline.shaderProgram)

    # # Setting up the clear screen color
    # glClearColor(0.15, 0.15, 0.15, 1.0)

    # # Creating shapes on GPU memory
    # shapeQuad = bs.createRainbowQuad()
    # gpuQuad = es.GPUShape().initBuffers()
    # pipeline.setupVAO(gpuQuad)
    # gpuQuad.fillBuffers(shapeQuad.vertices, shapeQuad.indices, GL_STATIC_DRAW)


    # initilize imgui context (see documentation)
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    # It is important to set the callback after the imgui setup
    glfw.set_key_callback(window, on_key)

    locationX = 0.0
    locationY = 0.0
    locationZ = 0.0
    locationRX = 0.0
    locationRY = 0.0
    locationRZ = 0.0
    scale = 1
    angle = 0.0

    while not glfw.window_should_close(window):

        impl.process_inputs()
        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))
        # Using GLFW to check for input events

        # Poll and handle events (inputs, window resize, etc.)
        # You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        # - When io.want_capture_mouse is true, do not dispatch mouse input data to your main application.
        # - When io.want_capture_keyboard is true, do not dispatch keyboard input data to your main application.
        # Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        # io = imgui.get_io()
        #print(io.want_capture_mouse, io.want_capture_keyboard)
        glfw.poll_events()

        # Filling or not the shapes depending on the controller state
        # if (controller.fillPolygon):
        #     glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        # else:
        #     glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if controller.showAxis:
            glUseProgram(mvpPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawCall(gpuAxis, GL_LINES)

        glUseProgram(pipeline.shaderProgram)
        # Rotación y traslacion del modelo
        sunRot = sg.findNode(solarSystem, "sunRotation")
        sunRot.transform = tr.matmul([tr.translate(locationX, locationY, locationZ), tr.rotationX(locationRX), tr.rotationY(locationRY), tr.rotationZ(locationRZ), tr.uniformScale(scale)])
        sg.drawSceneGraphNode(solarSystem, pipeline, "model")
        # imgui function
        impl.process_inputs()

        locationX, locationY, locationZ, locationRX, locationRY, locationRZ, angle, scale, color = \
            transformGuiOverlay(locationX, locationY, locationZ, locationRX, locationRY, locationRZ, angle, scale, color)

        # # Setting uniforms and drawing the Quad
        # glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "transform"), 1, GL_TRUE,
        #         tr.translate(locationX, locationY, 0.0),
        # )
        # glUseProgram(mvpPipeline.shaderProgram)
        # glUniform3f(glGetUniformLocation(mvpPipeline.shaderProgram, "modulationColor"),
        #     color[0], color[1], color[2])
        

        # Drawing the imgui texture over our drawing
        # glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        impl.render(imgui.get_draw_data())

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # # freeing GPU memory
    # gpuQuad.clear()

    # freeing GPU memory
    # gpuAxis.clear()
    # solarSystem.clear()

    impl.shutdown()
    glfw.terminate()