
/* - INCLUDES */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <assert.h>
#include <math.h>

#include "glad/glad.h"
#include "glfw/glfw3.h"
#include "cglm/struct.h"

#include "sh_vector.h"


/* - DEBUGGING */
#define DEBUG

#ifdef DEBUG
#   define LOG(...) printf(__VA_ARGS__)
#else
#   define LOG(...)
#endif



/* - CONSTANTS */

/* Memory */
#define BATCH_SIZE       1024
#define MAX_OBJECTS      16
#define MAX_VIEWERS      8
#define MAX_SEG_POINTS  (10 * MAX_OBJECTS)

/* Control */
#define SELECTION_DISTANCE 10

/* Style */
#define WINDOW_INIT_W   800
#define WINDOW_INIT_H   600
#define CTRLP_SIZE      3
#define CIRCLE_SEGMENTS 64

#define BGD_C         0xFFFFFFFF
vec4s   bgd_c;
#define SEGMENT_C     0x101010FF
vec4s   segment_c;
#define CTRLP_C       0x303030FF
vec4s   ctrlp_c;
#define SELECTION_C   0x1E81B0FF
vec4s   selection_c;

/* - STRUCTS */
typedef struct {
    vec2s position;
    vec4s color;
} Vertex;

typedef struct {
    GLint vao;
    GLint vb;
    GLint ib;
    size_t ic;
} VertexArray;

typedef struct {
    vec2s p1;
    vec2s p2;
} Segment;

typedef struct {
    vec2s position;
    vec2s size;
} Rect;

typedef struct {
    vec2s position;
    vec2s radius_pos;
} Circle;

typedef struct {
    vec2s position;
} Viewer;

SH_VECTOR_DECL(Segment,    MAX_OBJECTS,    segment);
SH_VECTOR_DECL(Rect,       MAX_OBJECTS,    rect);
SH_VECTOR_DECL(Circle,     MAX_OBJECTS,    circle);
SH_VECTOR_DECL(Viewer,     MAX_VIEWERS,    viewer);
SH_VECTOR_DECL(vec2s,      MAX_SEG_POINTS, vec2);
SH_VECTOR_DECL(float,      MAX_OBJECTS,    float);
SH_VECTOR_DECL(Vertex,     BATCH_SIZE,     vertex);
SH_VECTOR_DECL(uint32_t,   BATCH_SIZE,     index)

typedef struct {

    enum {
        SE_NONE = 0, 
        SE_SEG, 
        SE_RECT,
        SE_CIRCLE, 
        SE_VIEWER
    } type;

    enum {
        CP_NONE = 0,
        CP_LT,
        CP_RT,
        CP_LB,
        CP_RB,
        CP_P1,
        CP_P2,
        CP_RADIUS,
        CP_CENTER
    } control_point;

    size_t index;

} ControllableSelection;


/* - STATE */
struct _UIstate {

    sh_vector_vertex vertices;
    sh_vector_index  indices;

    GLint program;
    VertexArray vao;

    vec2s view_center;
    mat4s view_uniform;
    GLint view_uniform_location;

    GLenum mode;
} UIstate;

struct _WindowState {
    uint32_t    width;
    uint32_t    height;
    GLFWwindow *handle;
} WindowState;

struct _ControlState {

    vec2s click_pos;
    union {
        vec2s   view_center;
        Rect    rect;
        Circle  circle;
        Segment segment;
        Viewer  viewer;
    } reference;
    vec2s selection_offset;
    vec2s build_origin;

    int show_control_points;
    
    sh_vector_segment segments;
    sh_vector_rect    rects;
    sh_vector_circle  circles;
    sh_vector_viewer  viewers;

    ControllableSelection selection;

    enum {
        ST_IDLE = 0,
        ST_PAN,
        ST_BUILD_SEGMENT,
        ST_BUILD_RECT,
        ST_BUILD_CIRCLE,
        ST_MOVE
    } state;

} ControlState;

struct _Simulation {

    VertexArray vao;
    GLint       program;

    sh_vector_vec2  seg_points;
    sh_vector_vec2  circle_centers;
    sh_vector_float circle_radii;

    struct _SimulationUniformLocations {

        GLint 
            seg_points,
            seg_point_count,
            circle_centers,
            circle_radii,
            circle_count,
            viewer_origin,
            view_center,
            view,
            buffer_size;
    } uniform_locations;

} Simulation;



/* - SHADER SOURCES */

#define SHADER_SOURCE(...) #__VA_ARGS__

const char *generic_vertex_source = SHADER_SOURCE(#version 330 core\n

    layout (location = 0) in vec2 a_position;
    layout (location = 1) in vec4 a_color;

    uniform mat4 u_view;

    out vec4 v_color;

    void main() {

        v_color = a_color;
        gl_Position = u_view * vec4(a_position, 0, 1);
    }
);

const char *generic_fragment_source = SHADER_SOURCE(#version 330 core\n

    in vec4 v_color;

    void main() {

        gl_FragColor = v_color;
    }
);

const char *simulation_fragment_source = SHADER_SOURCE(#version 330 core\n

    uniform vec2  u_seg_points[256];
    uniform int   u_seg_point_count;
    uniform vec2  u_circle_centers[256];
    uniform float u_circle_radii[256];
    uniform int   u_circle_count;
    uniform vec2  u_viewer_origin;
    uniform vec2  u_view_center;
    uniform vec2  u_buffer_size;
    uniform int   u_sample_count;
    uniform float u_sample_diffusion;

    float crs2(vec2 a, vec2 b) 
    {
        return a.x * b.y - a.y * b.x;
    }

    bool line_intersect(vec2 p, vec2 p2, vec2 q, vec2 q2)
    {
        vec2 r = p2 - p;
        vec2 s = q2 - q;

        float crs = crs2(r, s);
        if (abs(crs) < 0.000001) return false;

        float t = crs2(q - p, s / crs);
        if (t < 0.0 || t > 1.0) return false;

        float u = crs2(q - p, r / crs);
        if (u < 0.0 || u > 1.0) return false;

        return true;
    }

    bool circle_intersect(vec2 p, vec2 q, vec2 center, float radius)
    {
        vec2 a = q - p;
        vec2 b = center - p;
        float f = dot(a, b) / length(a) / length(a);
        vec2 proj = f * a;
        float proj_d1 = distance(proj, b);
        float proj_d2 = length(proj) * sign(f);

        if (proj_d1 > radius) return false;

        float x = sqrt(radius * radius - proj_d1 * proj_d1);
        float x0 = proj_d2 - x;
        float x1 = proj_d2 + x;

        float l = length(a);

        if ((x0 < 0.0 || x0 > l) && (x1 < 0.0 || x1 > l)) return false;

        return true;
    }

    void main() {

        vec2 sample = vec2(gl_FragCoord.x - 0.5 * u_buffer_size.x, 0.5 * u_buffer_size.y - gl_FragCoord.y) + u_view_center;

        for (int i = 0; i < u_seg_point_count; i += 2) {

            if (line_intersect(
                u_seg_points[i],
                u_seg_points[i + 1],
                u_viewer_origin,
                sample
            )) {
                discard;
            }
        }

        for (int i = 0; i < u_circle_count; i++) {

            if (circle_intersect(
                u_viewer_origin,
                sample,
                u_circle_centers[i],
                u_circle_radii[i]
            )) {
                discard;
            }
        }

        gl_FragColor = vec4(0.2, 0.2, 0.2, 0.2);
    }
);



/* - FUNCTION DECLARATIONS */

/* OpenGL */
void  catchGLError(const char *expr, int line, const char *file);
#define GL_CALL(x) do { x; catchGLError(#x, __LINE__, __FILE__); } while (0)

GLint compileProgram(const char *vertex_source, const char *fragment_source);
GLint getUniformLocation(GLint program, const char *name);

VertexArray vaoCreate();
void vaoVertexData(VertexArray *vao, Vertex *data, size_t count);
void vaoIndexData(VertexArray *vao, uint32_t *data, size_t count);
void vaoDraw(VertexArray *vao, GLenum mode);

/* UI */
void initUIstate();

void pushUIline(float x1, float y1, float x2, float y2, vec4s color);
void pushUIquad(float x, float y, float w, float h, vec4s color);
void pushUIcircle(float x, float y, float radius, vec4s color);
void beginUI(GLenum mode);
void flushUI();

void updateUIView();
vec4s color(int hex);

void drawControlPoint(float x, float y, int selected);
void drawRectLines(Rect *rect, int selected);
void drawSegmentLines(Segment *seg, int selected);
void drawCircleLines(Circle *circle, int selected);
void drawRectControlPoints(Rect *rect, int selected);
void drawSegmentControlPoints(Segment *seg, int selected);
void drawCircleControlPoints(Circle *circle, int selected);
void drawControlState();

/* Simulation */
void initSimulation();
void renderViewers();
void pushSegments();
void pushCircles();

/* Callbacks */
void callbackResize(GLFWwindow *window, int width, int height);
void callbackMouse(GLFWwindow *window, int button, int action, int mods);
void callbackKey(GLFWwindow *window, int key, int scancode, int action, int mods);

/* Control State */
void initControlState();
void updateControlState();

int lineIsHovered(vec2s mouse, vec2s p1, vec2s p2);
int circleIsHovered(vec2s mouse, vec2s center, float radius);
int controlPointIsHovered(vec2s mouse, vec2s pos);

int testSegmentSelection(vec2s mouse, size_t index);
int testRectSelection(vec2s mouse, size_t index);
int testCircleSelection(vec2s mouse, size_t index);
int testViewerSelection(vec2s mouse, size_t index);
void testSelections();

vec2s mousePosW();
vec2s mousePos();
float getRadius(Circle *circle);

void moveSegment(vec2s move_offset);
void moveRect(vec2s move_offset);
void moveCircle(vec2s move_offset);
void moveViewer(vec2s move_offset);

void buildSegment();
void buildRect();
void buildCircle();

void deleteSelected();


/* - ENTRY POINT */
int main(int argc, char *argv[])
{

    glfwInit();

    glfwWindowHint(GLFW_SAMPLES, 16);
    WindowState = (struct _WindowState) {

        .handle = glfwCreateWindow(WINDOW_INIT_W, WINDOW_INIT_H, "Test Window", NULL, NULL),
        .width  = WINDOW_INIT_W,
        .height = WINDOW_INIT_H
    };
    assert(WindowState.handle);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    glfwSetWindowSizeCallback(WindowState.handle, callbackResize);
    glfwSetMouseButtonCallback(WindowState.handle, callbackMouse);
    glfwSetKeyCallback(WindowState.handle, callbackKey);

    glfwMakeContextCurrent(WindowState.handle);
    glfwSwapInterval(1);

    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);

    GL_CALL( glEnable(GL_BLEND) );
    GL_CALL( glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA) );  

    initUIstate();
    initControlState();
    initSimulation();

    while (!glfwWindowShouldClose(WindowState.handle))
    {
        glClear(GL_COLOR_BUFFER_BIT);
        glClearColor(bgd_c.r, bgd_c.g, bgd_c.b, 1.0f);

        pushSegments();
        pushCircles();
        renderViewers();

        beginUI(GL_TRIANGLES);

        drawControlState();

        flushUI();


        if (glfwGetKey(WindowState.handle, GLFW_KEY_RIGHT)      == GLFW_PRESS) { UIstate.view_center.x += 10.0f; }
        if (glfwGetKey(WindowState.handle, GLFW_KEY_LEFT)       == GLFW_PRESS) { UIstate.view_center.x -= 10.0f; }
        if (glfwGetKey(WindowState.handle, GLFW_KEY_DOWN)       == GLFW_PRESS) { UIstate.view_center.y += 10.0f; }
        if (glfwGetKey(WindowState.handle, GLFW_KEY_UP)         == GLFW_PRESS) { UIstate.view_center.y -= 10.0f; }
        if (glfwGetKey(WindowState.handle, GLFW_KEY_ESCAPE)     == GLFW_PRESS) { glfwSetWindowShouldClose(WindowState.handle, GLFW_TRUE); }

        glfwPollEvents();
        updateControlState();
        glfwSwapBuffers(WindowState.handle);
    }

    glfwTerminate();

    return 0;
}



/* - FUNCTION DEFINITIONS */

void catchGLError(const char *expr, int line, const char *file)
{

    GLenum e = glGetError();
    if (e == GL_NO_ERROR) return;

    printf("OpenGL errors\n\n");

    while (e != GL_NO_ERROR) {

        printf("\t%#04x\n", e);
        e = glGetError();
    }

    printf("\n\tgenerated when calling %s at %s(%d)\n", expr, file, line);
    getchar();
}

GLint compileProgram(const char *vertex_source, const char *fragment_source)
{

    /* Compile shaders */
    GLint vertex_shader, fragment_shader;
    GL_CALL( vertex_shader   = glCreateShader(GL_VERTEX_SHADER)   );
    GL_CALL( fragment_shader = glCreateShader(GL_FRAGMENT_SHADER) );

    GL_CALL( glShaderSource(vertex_shader,   1, &vertex_source,   0) );
    GL_CALL( glShaderSource(fragment_shader, 1, &fragment_source, 0) );

    GL_CALL( glCompileShader(vertex_shader)   );
    GL_CALL( glCompileShader(fragment_shader) );

    int status;
    char log[512];

    GL_CALL( glGetShaderiv(vertex_shader, GL_COMPILE_STATUS, &status) );
    if (!status) {

        GL_CALL( glGetShaderInfoLog(vertex_shader, 512, 0, log) );
        printf("Vertex Shader Compile Error:\n%s", log);
    }
    
    GL_CALL( glGetShaderiv(fragment_shader, GL_COMPILE_STATUS, &status) );
    if (!status) {

        GL_CALL( glGetShaderInfoLog(fragment_shader, 512, 0, log) );
        printf("Fragment Shader Compile Error:\n%s", log);
    }

    /* Link shaders */
    GLint program;
    GL_CALL( program = glCreateProgram() );

    GL_CALL( glAttachShader(program, vertex_shader)   );
    GL_CALL( glAttachShader(program, fragment_shader) );

    GL_CALL( glLinkProgram(program) );

    GL_CALL( glGetProgramiv(program, GL_LINK_STATUS, &status) );
    if (!status) {

        GL_CALL( glGetProgramInfoLog(program, 512, 0, log) );
        printf("Fragment Shader Compile Error:\n%s", log);
        getchar();
    }

    return program;
}

GLint getUniformLocation(GLint program, const char *name)
{
    GLint location;
    GL_CALL( location = glGetUniformLocation(program, name) );
    if (location < 0) { 
        
        printf("Uniform location not found: %s\n", name); 
        getchar(); 
    }

    return location;
}

VertexArray vaoCreate()
{
    VertexArray vao;

    GL_CALL( glGenVertexArrays(1, &vao.vao) );
    GL_CALL( glGenBuffers(1, &vao.vb) );
    GL_CALL( glGenBuffers(1, &vao.ib) );
    
    GL_CALL( glBindVertexArray(vao.vao) );
    GL_CALL( glBindBuffer(GL_ARRAY_BUFFER, vao.vb) );
    GL_CALL( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vao.ib) );

    GL_CALL( glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*)offsetof(Vertex, position)) );
    GL_CALL( glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const void*)offsetof(Vertex, color))    );

    GL_CALL( glEnableVertexAttribArray(0) );
    GL_CALL( glEnableVertexAttribArray(1) );

    vao.ic = 0;
    return vao;
}

void vaoVertexData(VertexArray *vao, Vertex *data, size_t count)
{
    GL_CALL( glBindBuffer(GL_ARRAY_BUFFER, vao->vb) );
    GL_CALL( glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * count, data, GL_STATIC_DRAW) );
}

void vaoIndexData(VertexArray *vao, uint32_t *data, size_t count)
{
    GL_CALL( glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vao->ib) );
    GL_CALL( glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * count, data, GL_STATIC_DRAW) );
    vao->ic = count;
}

void vaoDraw(VertexArray *vao, GLenum mode)
{
    GL_CALL( glBindVertexArray(vao->vao) );
    GL_CALL( glDrawElements(mode, (GLsizei)vao->ic, GL_UNSIGNED_INT, 0) );
}

void initUIstate()
{
    UIstate = (struct _UIstate) {
        .vertices = SH_VECTOR_INIT(vertex),
        .indices  = SH_VECTOR_INIT(index),
        .program  = compileProgram(generic_vertex_source, generic_fragment_source),
        .vao      = vaoCreate(),
        .mode     = GL_LINES
    };

    UIstate.view_uniform_location = getUniformLocation(UIstate.program, "u_view");
    UIstate.view_center = (vec2s) { WINDOW_INIT_W / 2, WINDOW_INIT_H / 2};
    updateUIView();

    bgd_c       = color(BGD_C);
    segment_c   = color(SEGMENT_C);
    ctrlp_c     = color(CTRLP_C);
    selection_c = color(SELECTION_C);

    GL_CALL( glEnable(GL_MULTISAMPLE) );
}

void pushUIline(float x1, float y1, float x2, float y2, vec4s color)
{
    if (
        UIstate.mode          != GL_LINES       || 
        UIstate.indices.size  >= BATCH_SIZE - 2 || 
        UIstate.vertices.size >= BATCH_SIZE - 2
    ) {

        flushUI();
        beginUI(GL_LINES);
    }

    size_t voffset = UIstate.vertices.size;

    sh_vector_vertex_push(&UIstate.vertices, (Vertex) {
        .position = (vec2s) { x1, y1 },
        .color = color
    });

    sh_vector_vertex_push(&UIstate.vertices, (Vertex) {
        .position = (vec2s) { x2, y2 },
        .color = color
    });

    sh_vector_index_push(&UIstate.indices, (uint32_t)voffset);
    sh_vector_index_push(&UIstate.indices, (uint32_t)voffset + 1);
}

void pushUIquad(float x, float y, float w, float h, vec4s color)
{
    if (
        UIstate.mode          != GL_TRIANGLES   || 
        UIstate.indices.size  >= BATCH_SIZE - 6 || 
        UIstate.vertices.size >= BATCH_SIZE - 4
    ) {

        flushUI();
        beginUI(GL_TRIANGLES);
    }

    size_t voffset = UIstate.vertices.size;

    sh_vector_vertex_push(&UIstate.vertices, (Vertex) {
        .position = (vec2s) { x, y },
        .color = color
    });

    sh_vector_vertex_push(&UIstate.vertices, (Vertex) {
        .position = (vec2s) { x + w, y },
        .color = color
    });

    sh_vector_vertex_push(&UIstate.vertices, (Vertex) {
        .position = (vec2s) { x, y + h },
        .color = color
    });

    sh_vector_vertex_push(&UIstate.vertices, (Vertex) {
        .position = (vec2s) { x + w, y + h },
        .color = color
    });

    sh_vector_index_push(&UIstate.indices, (uint32_t)voffset);
    sh_vector_index_push(&UIstate.indices, (uint32_t)voffset + 1);
    sh_vector_index_push(&UIstate.indices, (uint32_t)voffset + 2);
    sh_vector_index_push(&UIstate.indices, (uint32_t)voffset + 1);
    sh_vector_index_push(&UIstate.indices, (uint32_t)voffset + 2);
    sh_vector_index_push(&UIstate.indices, (uint32_t)voffset + 3);
}

void pushUIcircle(float x, float y, float radius, vec4s color)
{
    for (int i = 0; i < CIRCLE_SEGMENTS; i++) {

        float a0 = i * 2.0f * (float)M_PI / CIRCLE_SEGMENTS;
        float x0 = x + radius * cosf(a0);
        float y0 = y + radius * sinf(a0);

        float a1 = (i + 1) * 2.0f * (float)M_PI / CIRCLE_SEGMENTS;
        float x1 = x + radius * cosf(a1);
        float y1 = y + radius * sinf(a1);

        pushUIline(x0, y0, x1, y1, color);
    }
}

void beginUI(GLenum mode)
{
    sh_vector_vertex_clear(&UIstate.vertices);
    sh_vector_index_clear(&UIstate.indices);
    UIstate.mode = mode;
}

void flushUI()
{
    GL_CALL(glUseProgram(UIstate.program));
    updateUIView();
    GL_CALL( glUniformMatrix4fv(UIstate.view_uniform_location, 1, GL_FALSE, (const GLfloat*)UIstate.view_uniform.raw) );
    vaoVertexData(&UIstate.vao, UIstate.vertices.data, UIstate.vertices.size);
    vaoIndexData(&UIstate.vao,  UIstate.indices.data,  UIstate.indices.size);
    vaoDraw(&UIstate.vao, UIstate.mode);
}

void updateUIView()
{

    UIstate.view_uniform = glms_ortho(
        UIstate.view_center.x - WindowState.width  / 2 + (WindowState.width  % 2),
        UIstate.view_center.x + WindowState.width  / 2 - (WindowState.width  % 2),
        UIstate.view_center.y + WindowState.height / 2 - (WindowState.height % 2),
        UIstate.view_center.y - WindowState.height / 2 + (WindowState.height % 2),
        -1.0f, 
        100.0f
    );
}

void initControlState()
{
    ControlState = (struct _ControlState) {
        .segments  =  SH_VECTOR_INIT(segment),
        .rects     =  SH_VECTOR_INIT(rect),
        .circles   =  SH_VECTOR_INIT(circle),
        .viewers   =  SH_VECTOR_INIT(viewer),
        .selection = { .type = SE_NONE, .index = 0},
        .state     = ST_IDLE,
        .show_control_points = 1
    };
}

void callbackResize(GLFWwindow *window, int width, int height)
{

    WindowState.width = width;
    WindowState.height = height;

    GL_CALL( glViewport(0, 0, width, height) );
}

void callbackMouse(GLFWwindow *window, int button, int action, int mods)
{

    switch (action) {

        case GLFW_PRESS:
            switch (button)
            {
                case GLFW_MOUSE_BUTTON_LEFT:
                    ControlState.click_pos = mousePos();
                    testSelections();
                    if (ControlState.selection.type == SE_NONE) {
                        ControlState.state = ST_PAN;
                        ControlState.click_pos = mousePosW();
                        ControlState.reference.view_center = UIstate.view_center;
                    }
                    break;

                case GLFW_MOUSE_BUTTON_RIGHT:

                    ControlState.build_origin = mousePos();
                    if (glfwGetKey(WindowState.handle, GLFW_KEY_LEFT_SHIFT) == GLFW_TRUE) {

                        Rect rect = (Rect) { .position = ControlState.build_origin, .size = GLMS_VEC2_ZERO };
                        if (!sh_vector_rect_push(&ControlState.rects, rect)) break;

                        ControlState.selection = (ControllableSelection) {
                            .type = SE_RECT,
                            .index = ControlState.rects.size - 1,
                            .control_point = CP_NONE
                        };

                        ControlState.state = ST_BUILD_RECT;

                    } else if (glfwGetKey(WindowState.handle, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {

                        Viewer viewer = (Viewer) { .position = ControlState.build_origin };
                        if (!sh_vector_viewer_push(&ControlState.viewers, viewer)) break;

                        ControlState.selection = (ControllableSelection) {
                            .type = SE_VIEWER,
                            .index = ControlState.viewers.size - 1,
                            .control_point = CP_NONE 
                        };

                        ControlState.state = ST_IDLE;

                    } else if (glfwGetKey(WindowState.handle, GLFW_KEY_LEFT_ALT) == GLFW_PRESS) {
                        
                        Circle circle = (Circle) { .position = ControlState.build_origin, .radius_pos = ControlState.build_origin };
                        if (!sh_vector_circle_push(&ControlState.circles, circle)) break;

                        ControlState.selection = (ControllableSelection) {
                            .type = SE_CIRCLE,
                            .index = ControlState.circles.size - 1,
                            .control_point = CP_NONE
                        };

                        ControlState.state = ST_BUILD_CIRCLE;

                    } else {
                        
                        Segment seg = (Segment) { .p1 = ControlState.build_origin, .p2 = ControlState.build_origin };
                        if (!sh_vector_segment_push(&ControlState.segments, seg )) break;

                        ControlState.selection = (ControllableSelection) {
                            .type = SE_SEG,
                            .index = ControlState.segments.size - 1,
                            .control_point = CP_NONE
                        };

                        ControlState.state = ST_BUILD_SEGMENT;
                    }
            }
            break;

        case GLFW_RELEASE:
            ControlState.state = ST_IDLE;
            break;
    }
}

void callbackKey(GLFWwindow *window, int key, int scancode, int action, int mods)
{

    switch (action) {

        case GLFW_PRESS:
            switch (key) {
                
                case GLFW_KEY_DELETE:
                case GLFW_KEY_BACKSPACE:
                case GLFW_KEY_X:
                    deleteSelected();
                    break;

                case GLFW_KEY_H:
                    ControlState.show_control_points ^= 1;
                    break;

                default: break;
            }
            break;

        default: break;
    }
}

vec4s color(int hex)
{
    return (vec4s) {
        ((hex >> 24) & 0xff) / 255.99f,
        ((hex >> 16) & 0xff) / 255.99f,
        ((hex >> 8 ) & 0xff) / 255.99f,
        ((hex >> 0 ) & 0xff) / 255.99f
    };
}

void drawRectLines(Rect *rect, int selected)
{
    vec4s color = selected ? selection_c : segment_c;

    pushUIline(
        rect->position.x, 
        rect->position.y, 
        rect->position.x + rect->size.x, 
        rect->position.y, 
        color
    );

    pushUIline(
        rect->position.x + rect->size.x, 
        rect->position.y, 
        rect->position.x + rect->size.x, 
        rect->position.y + rect->size.y, 
        color
    );

    pushUIline(
        rect->position.x + rect->size.x, 
        rect->position.y + rect->size.y, 
        rect->position.x, 
        rect->position.y + rect->size.y, 
        color
    );

    pushUIline(
        rect->position.x, 
        rect->position.y + rect->size.y, 
        rect->position.x, 
        rect->position.y, 
        color
    );
}

void drawSegmentLines(Segment *seg, int selected)
{

    vec4s color = selected ? selection_c : segment_c;
    pushUIline(seg->p1.x, seg->p1.y, seg->p2.x, seg->p2.y, color);
}

void drawCircleLines(Circle *circle, int selected)
{

    vec4s color = selected ? selection_c : segment_c;
    pushUIcircle(circle->position.x, circle->position.y, getRadius(circle), color);
}

void drawControlPoint(float x, float y, int selected)
{
    vec4s color = selected ? selection_c : ctrlp_c;
    pushUIquad(x - CTRLP_SIZE, y - CTRLP_SIZE, 2 * CTRLP_SIZE, 2 * CTRLP_SIZE, color);
}

void drawRectControlPoints(Rect *rect, int selected)
{

    drawControlPoint(rect->position.x,                rect->position.y, selected);
    drawControlPoint(rect->position.x + rect->size.x, rect->position.y, selected);
    drawControlPoint(rect->position.x,                rect->position.y + rect->size.y, selected);
    drawControlPoint(rect->position.x + rect->size.x, rect->position.y + rect->size.y, selected);
    drawControlPoint(rect->position.x + rect->size.x / 2, rect->position.y + rect->size.y / 2, selected);
}

void drawSegmentControlPoints(Segment *seg, int selected)
{

    drawControlPoint(seg->p1.x, seg->p1.y, selected);
    drawControlPoint(seg->p2.x, seg->p2.y, selected);
    drawControlPoint((seg->p1.x + seg->p2.x) / 2, (seg->p1.y + seg->p2.y) / 2, selected);
}

void drawCircleControlPoints(Circle *circle, int selected)
{

    drawControlPoint(circle->position.x, circle->position.y, selected);
    drawControlPoint(circle->radius_pos.x, circle->radius_pos.y, selected);
}

void drawControlState()
{

    for (int i = 0; i < ControlState.segments.size; i++) {

        drawSegmentLines(&ControlState.segments.data[i], 
            i == ControlState.selection.index && ControlState.selection.type == SE_SEG);
    }

    for (int i = 0; i < ControlState.rects.size; i++) {

        drawRectLines(&ControlState.rects.data[i], 
            i == ControlState.selection.index && ControlState.selection.type == SE_RECT);
    }

    for (int i = 0; i < ControlState.circles.size; i++) {

        drawCircleLines(&ControlState.circles.data[i],
            i == ControlState.selection.index && ControlState.selection.type == SE_CIRCLE);
    }


    for (int i = 0; i < ControlState.viewers.size; i++) {

        drawControlPoint(ControlState.viewers.data[i].position.x, ControlState.viewers.data[i].position.y, 
            i == ControlState.selection.index && ControlState.selection.type == SE_VIEWER);
    }

    if (!ControlState.show_control_points) return;


    for (int i = 0; i < ControlState.segments.size; i++) {

        drawSegmentControlPoints(&ControlState.segments.data[i],
            i == ControlState.selection.index && ControlState.selection.type == SE_SEG);
    }

    for (int i = 0; i < ControlState.rects.size; i++) {

        drawRectControlPoints(&ControlState.rects.data[i],
            i == ControlState.selection.index && ControlState.selection.type == SE_RECT);
    }

    for (int i = 0; i < ControlState.circles.size; i++) {

        drawCircleControlPoints(&ControlState.circles.data[i],
            i == ControlState.selection.index && ControlState.selection.type == SE_CIRCLE);
    }

}

int lineIsHovered(vec2s mouse, vec2s p1, vec2s p2)
{

    /* Project mouse position to segment */
    vec2s s     = glms_vec2_sub(p2, p1);
    vec2s m     = glms_vec2_sub(mouse, p1);
    float dot   = glms_vec2_dot(s, m);
    float slen2 = glms_vec2_norm2(s);

    /* Check if projected point is outside segment */
    if (dot < 0 || dot > slen2) {
        return 0;
    }

    if (glms_vec2_norm(glms_vec2_sub(glms_vec2_scale(s, dot / slen2), m)) > SELECTION_DISTANCE) {
        return 0;
    }

    return 1;
}

int circleIsHovered(vec2s mouse, vec2s center, float radius)
{
    float d = glms_vec2_norm(glms_vec2_sub(mouse, center)) - radius;
    if (fabsf(d) < SELECTION_DISTANCE) {

        return 1;
    }

    return 0;
}

int controlPointIsHovered(vec2s mouse, vec2s pos)
{
    return glms_vec2_norm(glms_vec2_sub(mouse, pos)) < SELECTION_DISTANCE;
}

int testSegmentSelection(vec2s mouse, size_t index)
{   
    Segment *segment = &ControlState.segments.data[index];
    vec2s point;

    if (!ControlState.show_control_points) goto just_check_line;

    if (controlPointIsHovered(mouse, segment->p1)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_P1,
            .index         = index,
            .type          = SE_SEG,
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(segment->p1, mouse);
        ControlState.reference.segment = *segment;
        return 1;
    }

    if (controlPointIsHovered(mouse, segment->p2)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_P2,
            .index         = index,
            .type          = SE_SEG,
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(segment->p2, mouse);
        ControlState.reference.segment = *segment;
        return 1;
    }

    point = glms_vec2_scale(glms_vec2_add(segment->p1, segment->p2), 0.5f);
    if (controlPointIsHovered(mouse, point)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_CENTER,
            .index         = index,
            .type          = SE_SEG
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(point, mouse);
        ControlState.reference.segment = *segment;
        return 1;
    }

    just_check_line:

    if (lineIsHovered(mouse, segment->p1, segment->p2)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_NONE,
            .index         = index,
            .type          = SE_SEG
        };
        ControlState.state   = ST_IDLE;
        return 1;
    }

    return 0;
}

int testRectSelection(vec2s mouse, size_t index)
{
    Rect *rect = &ControlState.rects.data[index];
    vec2s point;

    if (!ControlState.show_control_points) goto just_check_line;
    
    if (controlPointIsHovered(mouse, rect->position)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_LT,
            .index         = index,
            .type          = SE_RECT
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(rect->position, mouse);
        ControlState.reference.rect = *rect;
        return 1;
    }
    
    point = (vec2s) { rect->position.x + rect->size.x, rect->position.y };
    if (controlPointIsHovered(mouse, point)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_RT,
            .index         = index,
            .type          = SE_RECT
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(point, mouse);
        ControlState.reference.rect = *rect;
        return 1;
    }
    
    point = (vec2s) { rect->position.x, rect->position.y + rect->size.y };
    if (controlPointIsHovered(mouse, point)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_LB,
            .index         = index,
            .type          = SE_RECT
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(point, mouse);
        ControlState.reference.rect = *rect;
        return 1;
    }
    
    point = glms_vec2_add(rect->position, rect->size);
    if (controlPointIsHovered(mouse, point)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_RB,
            .index         = index,
            .type          = SE_RECT
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(point, mouse);
        ControlState.reference.rect = *rect;
        return 1;
    }
    
    point = glms_vec2_add(rect->position, glms_vec2_scale(rect->size, 0.5f));
    if (controlPointIsHovered(mouse, point)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_CENTER,
            .index         = index,
            .type          = SE_RECT
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(point, mouse);
        ControlState.reference.rect = *rect;
        return 1;
    }

    just_check_line:

    if (
        lineIsHovered(mouse, 
            (vec2s) { rect->position.x,                rect->position.y }, 
            (vec2s) { rect->position.x + rect->size.x, rect->position.y }) ||
        lineIsHovered(mouse, 
            (vec2s) { rect->position.x + rect->size.x, rect->position.y }, 
            (vec2s) { rect->position.x + rect->size.x, rect->position.y + rect->size.y }) ||
        lineIsHovered(mouse, 
            (vec2s) { rect->position.x + rect->size.x, rect->position.y + rect->size.y }, 
            (vec2s) { rect->position.x,                rect->position.y + rect->size.y }) ||
        lineIsHovered(mouse, 
            (vec2s) { rect->position.x,                rect->position.y + rect->size.y }, 
            (vec2s) { rect->position.x,                rect->position.y })
    ) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_NONE,
            .index         = index,
            .type          = SE_RECT
        };
        ControlState.state = ST_IDLE;
        return 1;
    }

    return 0;
}

int testCircleSelection(vec2s mouse, size_t index)
{
    Circle *circle = &ControlState.circles.data[index];

    if (!ControlState.show_control_points) goto just_check_line;

    if (controlPointIsHovered(mouse, circle->position)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_CENTER,
            .index         = index,
            .type          = SE_CIRCLE,
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(circle->position, mouse);
        ControlState.reference.circle = *circle;
        return 1;
    }

    if (controlPointIsHovered(mouse, circle->radius_pos)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_RADIUS,
            .index         = index,
            .type          = SE_CIRCLE,
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(circle->radius_pos, mouse);
        ControlState.reference.circle = *circle;
        return 1;
    }

    just_check_line:

    if (circleIsHovered(mouse, circle->position, getRadius(circle))) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_NONE,
            .index         = index,
            .type          = SE_CIRCLE
        };
        ControlState.state   = ST_IDLE;
        return 1;
    }

    return 0;
}

int testViewerSelection(vec2s mouse, size_t index)
{
    Viewer *viewer = &ControlState.viewers.data[index];
    
    if (controlPointIsHovered(mouse, viewer->position)) {

        ControlState.selection = (ControllableSelection) {
            .control_point = CP_NONE,
            .index         = index,
            .type          = SE_VIEWER
        };
        ControlState.state = ST_MOVE;
        ControlState.selection_offset = glms_vec2_sub(viewer->position, mouse);
        ControlState.reference.viewer = *viewer;
        return 1;
    }

    return 0;
}

void testSelections()
{

    vec2s mouse = mousePos();

    for (int i = 0; i < ControlState.viewers.size; i++) {

        if (testViewerSelection(mouse, i)) {
            return;
        }
    }

    for (int i = 0; i < ControlState.segments.size; i++) {

        if (testSegmentSelection(mouse, i)) {
            return;
        }
    }

    for (int i = 0; i < ControlState.rects.size; i++) {

        if (testRectSelection(mouse, i)) {
            return;
        }
    }

    for (int i = 0; i < ControlState.circles.size; i++) {

        if (testCircleSelection(mouse, i)) {
            return;
        }
    }

    ControlState.selection = (ControllableSelection) {0};
}

vec2s mousePosW()
{
    double x, y;
    glfwGetCursorPos(WindowState.handle, &x, &y);
    return (vec2s) { (float)x, (float)y };
}

vec2s mousePos()
{

    vec2s m = mousePosW();
    m.x -= WindowState.width / 2;
    m.y -= WindowState.height / 2;
    m = glms_vec2_add(m, UIstate.view_center);
    return m;
}

float getRadius(Circle *circle)
{

    return glms_vec2_norm(glms_vec2_sub(circle->position, circle->radius_pos));
}

void updateControlState()
{
    vec2s move_offset;

    switch (ControlState.state) {

        case ST_IDLE:
            break;

        case ST_PAN:
            move_offset = glms_vec2_sub(mousePosW(), ControlState.click_pos);
            UIstate.view_center = glms_vec2_sub(ControlState.reference.view_center, move_offset);
            break;
        
        case ST_MOVE:

            move_offset = glms_vec2_sub(mousePos(), ControlState.click_pos);
            switch (ControlState.selection.type) {

                case SE_SEG:

                    moveSegment(move_offset);
                    break;

                case SE_RECT:

                    moveRect(move_offset);
                    break;

                case SE_CIRCLE:

                    moveCircle(move_offset);
                    break;

                case SE_VIEWER:

                    moveViewer(move_offset);
                    break;

                default: break;
            }
            break;
        
        case ST_BUILD_RECT:

            buildRect();
            break;

        case ST_BUILD_SEGMENT:

            buildSegment();
            break;

        case ST_BUILD_CIRCLE:

            buildCircle();
            break;

        default: break;
    }
}

void moveSegment(vec2s move_offset)
{
    Segment *seg = &ControlState.segments.data[ControlState.selection.index];
    Segment *ref = &ControlState.reference.segment;

    switch (ControlState.selection.control_point) {

        case CP_P1:
            seg->p1 = glms_vec2_add(glms_vec2_add(ref->p1, move_offset), ControlState.selection_offset);
            break;

        case CP_P2:
            seg->p2 = glms_vec2_add(glms_vec2_add(ref->p2, move_offset), ControlState.selection_offset);
            break;

        case CP_CENTER:
            seg->p1 = glms_vec2_add(glms_vec2_add(ref->p1, move_offset), ControlState.selection_offset);
            seg->p2 = glms_vec2_add(glms_vec2_add(ref->p2, move_offset), ControlState.selection_offset);
            break;

        default: break;
    }
}

void moveRect(vec2s move_offset)
{
    Rect *rect = &ControlState.rects.data[ControlState.selection.index];
    Rect *ref  = &ControlState.reference.rect;

    switch (ControlState.selection.control_point) {

        case CP_LT:
            rect->position = glms_vec2_add(glms_vec2_add(ref->position, move_offset), ControlState.selection_offset);
            rect->size     = glms_vec2_sub(ref->size, glms_vec2_add(move_offset, ControlState.selection_offset));
            break;
        
        case CP_RT:
            rect->size.x     = ref->size.x     + move_offset.x + ControlState.selection_offset.x;
            rect->position.y = ref->position.y + move_offset.y + ControlState.selection_offset.y;

            rect->size.y     = ref->size.y     - move_offset.y - ControlState.selection_offset.y;
            break;
            
        case CP_LB:
            rect->size.y     = ref->size.y     + move_offset.y + ControlState.selection_offset.y;
            rect->position.x = ref->position.x + move_offset.x + ControlState.selection_offset.x;

            rect->size.x     = ref->size.x     - move_offset.x - ControlState.selection_offset.x;
            break;
            
        case CP_RB:
            rect->size = glms_vec2_add(glms_vec2_add(ref->size, move_offset), ControlState.selection_offset);
            break;
            
        case CP_CENTER:
            rect->position = glms_vec2_add(glms_vec2_add(ref->position, move_offset), ControlState.selection_offset);
            break;

        default: break;
    }
}

void moveCircle(vec2s move_offset)
{

    Circle *circle = &ControlState.circles.data[ControlState.selection.index];
    Circle *ref = &ControlState.reference.circle;

    switch (ControlState.selection.control_point) {

        case CP_CENTER:
            circle->position   = glms_vec2_add(glms_vec2_add(ref->position, move_offset), ControlState.selection_offset);
            circle->radius_pos = glms_vec2_add(glms_vec2_add(ref->radius_pos, move_offset), ControlState.selection_offset);
            break;

        case CP_RADIUS:
            circle->radius_pos = glms_vec2_add(glms_vec2_add(ref->radius_pos, move_offset), ControlState.selection_offset);
            break;

        default: break;
    }
}

void moveViewer(vec2s move_offset)
{

    Viewer *viewer = &ControlState.viewers.data[ControlState.selection.index];
    Viewer *ref    = &ControlState.reference.viewer;

    viewer->position = glms_vec2_add(glms_vec2_add(ref->position, move_offset), ControlState.selection_offset);
}

void buildSegment()
{

    Segment *seg = &ControlState.segments.data[ControlState.selection.index];
    seg->p2 = mousePos();
}

void buildRect()
{

    Rect *rect = &ControlState.rects.data[ControlState.selection.index];
    rect->size = glms_vec2_sub(mousePos(), ControlState.build_origin);
}

void buildCircle()
{

    Circle *circle = &ControlState.circles.data[ControlState.selection.index];
    circle->radius_pos = mousePos();
}

void deleteSelected()
{

    if (ControlState.selection.type == SE_NONE) return;

    switch (ControlState.selection.type) {

        case SE_RECT:
            sh_vector_rect_remove(&ControlState.rects, ControlState.selection.index);
            break;

        case SE_SEG:
            sh_vector_segment_remove(&ControlState.segments, ControlState.selection.index);
            break;

        case SE_CIRCLE:
            sh_vector_circle_remove(&ControlState.circles, ControlState.selection.index);
            break;

        case SE_VIEWER:
            sh_vector_viewer_remove(&ControlState.viewers, ControlState.selection.index);
            break;

        default: break;
    }

    ControlState.selection.type = SE_NONE;
}

void pushSegments()
{

    for (int i = 0; i < ControlState.rects.size; i++) {

        Rect *rect = &ControlState.rects.data[i];
        vec2s p0 = rect->position;
        vec2s p1 = (vec2s) { rect->position.x + rect->size.x, rect->position.y                };
        vec2s p2 = (vec2s) { rect->position.x + rect->size.x, rect->position.y + rect->size.y };
        vec2s p3 = (vec2s) { rect->position.x,                rect->position.y + rect->size.y };

        sh_vector_vec2_push(&Simulation.seg_points, p0);
        sh_vector_vec2_push(&Simulation.seg_points, p1);

        sh_vector_vec2_push(&Simulation.seg_points, p1);
        sh_vector_vec2_push(&Simulation.seg_points, p2);

        sh_vector_vec2_push(&Simulation.seg_points, p2);
        sh_vector_vec2_push(&Simulation.seg_points, p3);

        sh_vector_vec2_push(&Simulation.seg_points, p3);
        sh_vector_vec2_push(&Simulation.seg_points, p0);
    }

    for (int i = 0; i < ControlState.segments.size; i++) {

        Segment *seg = &ControlState.segments.data[i];

        sh_vector_vec2_push(&Simulation.seg_points, seg->p1);
        sh_vector_vec2_push(&Simulation.seg_points, seg->p2);
    }

    GL_CALL( glUseProgram(Simulation.program) );
    GL_CALL( glUniform2fv(Simulation.uniform_locations.seg_points, (GLsizei)Simulation.seg_points.size, (const GLfloat*)&Simulation.seg_points.data) );
    GL_CALL( glUniform1i(Simulation.uniform_locations.seg_point_count, (GLint)Simulation.seg_points.size) );

    sh_vector_vec2_clear(&Simulation.seg_points);
}

void pushCircles()
{

    for (int i = 0; i < ControlState.circles.size; i++) {

        Circle *circle = &ControlState.circles.data[i];

        sh_vector_vec2_push(&Simulation.circle_centers, circle->position);
        sh_vector_float_push(&Simulation.circle_radii, getRadius(circle));
    }

    GL_CALL( glUseProgram(Simulation.program) );
    GL_CALL( glUniform2fv(Simulation.uniform_locations.circle_centers, (GLsizei)Simulation.circle_centers.size, (const GLfloat*)&Simulation.circle_centers.data) );
    GL_CALL( glUniform1fv(Simulation.uniform_locations.circle_radii,   (GLsizei)Simulation.circle_radii.size,   (const GLfloat*)&Simulation.circle_radii.data) );
    GL_CALL( glUniform1i(Simulation.uniform_locations.circle_count,    (GLint)Simulation.circle_centers.size) );

    sh_vector_vec2_clear(&Simulation.circle_centers);
    sh_vector_float_clear(&Simulation.circle_radii);
}


void initSimulation()
{

    Simulation.vao = vaoCreate();
    Simulation.program = compileProgram(generic_vertex_source, simulation_fragment_source);

    Simulation.uniform_locations = (struct _SimulationUniformLocations) {
        .seg_points       = getUniformLocation(Simulation.program, "u_seg_points"),
        .seg_point_count  = getUniformLocation(Simulation.program, "u_seg_point_count"),
        .circle_centers   = getUniformLocation(Simulation.program, "u_circle_centers"),
        .circle_radii     = getUniformLocation(Simulation.program, "u_circle_radii"),
        .circle_count     = getUniformLocation(Simulation.program, "u_circle_count"),
        .view             = getUniformLocation(Simulation.program, "u_view"),
        .viewer_origin    = getUniformLocation(Simulation.program, "u_viewer_origin"),
        .view_center      = getUniformLocation(Simulation.program, "u_view_center"),
        .buffer_size      = getUniformLocation(Simulation.program, "u_buffer_size")
    };

    Vertex vdata[] = {
        { { -1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } },
        { {  1.0f, -1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } },
        { { -1.0f,  1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } },
        { {  1.0f,  1.0f }, { 1.0f, 1.0f, 1.0f, 1.0f } }
    };

    uint32_t idata[] = {
        0, 1, 2,
        1, 2, 3
    };

    vaoVertexData(&Simulation.vao, vdata, sizeof(vdata) / sizeof(vdata[0]));
    vaoIndexData(&Simulation.vao,  idata, sizeof(idata) / sizeof(idata[0]));

    Simulation.seg_points       = SH_VECTOR_INIT(vec2);
    Simulation.circle_centers   = SH_VECTOR_INIT(vec2);
    Simulation.circle_radii     = SH_VECTOR_INIT(float);
}

void renderViewers()
{

    GL_CALL( glUseProgram(Simulation.program) );

    mat4s basic_view = GLMS_MAT4_IDENTITY;
    GL_CALL( glUniformMatrix4fv(Simulation.uniform_locations.view, 1, GL_FALSE, (const GLfloat*)basic_view.raw) );
    
    GL_CALL( glUniform2f(Simulation.uniform_locations.buffer_size, (GLfloat)WindowState.width, (GLfloat)WindowState.height) );
    GL_CALL( glUniform2f(Simulation.uniform_locations.view_center, UIstate.view_center.x, UIstate.view_center.y) );

    for (int i = 0; i < ControlState.viewers.size; i++) {

        Viewer *viewer = &ControlState.viewers.data[i];
        GL_CALL( glUniform2f(Simulation.uniform_locations.viewer_origin, viewer->position.x, viewer->position.y) );
        vaoDraw(&Simulation.vao, GL_TRIANGLES);
    }
}