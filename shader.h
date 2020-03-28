#ifndef __SHADER_H__
#define __SHADER_H__

#ifdef OSG_GLES2_AVAILABLE
	#define VERSION_COMMON \
	"#version 100\n"
#else
	#define VERSION_COMMON \
	"#version 130\n" \
	"#define highp\n" \
	"#define mediump\n" \
	"#define lowp\n"
#endif

#ifdef ANDROID
	#define VERTEX_COMMON \
	"#if __VERSION__ >= 130\n" \
	"   #define attribute in\n" \
	"   #define varying out\n" \
	"#endif\n" \
	"attribute vec4 osg_Vertex;\n" /*shows proposed pixel position(xyz) in image*/ \
	"attribute vec4 osg_Color;\n" \
	"attribute vec4 osg_MultiTexCoord0;\n" /*shows normal pixel coordinates in tile*/ \
	"uniform mat4 osg_ModelViewProjectionMatrix;\n" /*shows rotation from parent nodes*/ \
	"uniform mat4 osg_ModelViewMatrix;\n"
#else //desktop linux
	#define VERTEX_COMMON \
	"#if __VERSION__ >= 130\n" \
	"   #define attribute in\n" \
	"   #define varying out\n" \
	"#endif\n" \
	"#define osg_Vertex gl_Vertex\n" \
	"#define osg_Color gl_Color\n" \
	"#define osg_MultiTexCoord0 gl_MultiTexCoord0\n" \
	"#define osg_ModelViewProjectionMatrix gl_ModelViewProjectionMatrix\n" \
	"#define osg_ModelViewMatrix gl_ModelViewMatrix\n" //definition causes warning, but works!!!

#endif

#define FRAGMENT_COMMON \
	"#if __VERSION__ >= 130\n" \
	"   #define varying in\n" \
	"   #define texture2D texture\n" \
	"   out vec4 mgl_FragColor;\n" \
	"#else\n" \
	"   #define mgl_FragColor gl_FragColor\n" \
	"#endif\n"

//ints & floats "minimum" ranges:
//ints:
//	lowp [-255, +255]
//	mediump [-1023, +1023]
//	highp [-65535, +65535] -implementation is optional(but implemented in Galaxy S2)!!!! (#if GL_FRAGMENT_PRECISION_HIGH == 1)
//floats:
//	lowp - 8 bit, floating point range: -2 to 2, integer range: -2^8 to 2^8
//	mediump - 10 bit, floating point range: -2^14 to 2^14, integer range: -2^10 to 2^10
//	highp - 16-bit, floating point range: -2^62 to 2^62, integer range: -2^16 to 2^16 -implementation is optional!!!!

//defaults in vertex shader:
//	precision highp float;
//	precision highp int;
//defaults in fragment shader:
//	precision mediump int;

#ifdef OSG_GLES2_AVAILABLE
	//in vertex, defaults are already highp
	#define FRAGMENT_MEDIUMP \
	"precision mediump float;\n"
	#define FRAGMENT_HIGHP  \
	"#if GL_FRAGMENT_PRECISION_HIGH == 1\n" \
	"	precision highp float;\n" \
	"#else\n" \
	"	precision mediump float;\n" \
	"#endif\n"
#else
//precision qualifiers not supported in full opengl
	#define FRAGMENT_MEDIUMP ""
	#define FRAGMENT_HIGHP ""
#endif

//1st stage shader, color part/////////////////////////////////////////////////////////////////////

static char imageVertexShader[] =
	VERSION_COMMON
	VERTEX_COMMON
	"varying vec4 texCoord;\n"
	"\n"
	"void main(void) {\n"
	"   gl_Position = osg_ModelViewProjectionMatrix * osg_Vertex;\n"
	"   texCoord = osg_MultiTexCoord0;\n"
	"}\n";

static char imageFragmentShader[] =
	VERSION_COMMON
	FRAGMENT_COMMON
	FRAGMENT_MEDIUMP
	"uniform sampler2D baseTexture;\n"
	"varying vec4 texCoord;\n"
	"void main() {\n"
	"	vec4 tex = texture2D (baseTexture, texCoord.xy);\n" //no space after texture2D will cause warning
	"	mgl_FragColor = vec4(tex.r, tex.g, tex.b, tex.a);\n"
	"}\n";

//1st stage shader, depth part/////////////////////////////////////////////////////////////////////

static char depthVertexShader[] =
	VERSION_COMMON
	VERTEX_COMMON
	"varying vec4 texCoord;\n"
	"varying vec4 cs_position;\n"
	"\n"
	"void main(void) {\n"
	"   gl_Position = osg_ModelViewProjectionMatrix * osg_Vertex;\n"
	"   texCoord = osg_MultiTexCoord0;\n"
	"   cs_position = osg_ModelViewMatrix * osg_Vertex;\n" //in camera coordinates
	"}\n";

static char depthFragmentShader[] =
	VERSION_COMMON
	FRAGMENT_COMMON
	FRAGMENT_HIGHP
	"uniform sampler2D baseTexture;\n"
	"varying vec4 texCoord;\n" //varyings are interpolated(in osg::Texture2D::LINEAR) between vertices
	"varying vec4 cs_position;\n"
	"void main() {\n"
	"	float originalZ = -cs_position.z;\n"	//in vertex shader: "	cs_position = osg_ModelViewMatrix * osg_Vertex;\n" //in camera coordinates
		//direct distance, contour solution
		//use (1., 1., 1.) as infinity space
		// max = 255.*255. = 65 km, resolution = 1/255m = 4mm
	"	mgl_FragColor = vec4(floor(floor(originalZ)/255.)/255., fract(floor(originalZ)/255.), fract(originalZ), 1.);\n"  //verified in osg::Texture2D::NEAREST
		//recovery: color.r*(255.*255.) + color.g*255. + color.b //verified, (colors within [0,1])
	"}\n";

//2nd stage shaders/////////////////////////////////////////////////////////////////////////////

static char lensVertexShader[] =
	VERSION_COMMON
	VERTEX_COMMON
	"varying vec4 distortedTexCoord;\n"
	"\n"
	"void main(void) {\n"
	"   gl_Position = osg_ModelViewProjectionMatrix * osg_Vertex;\n"
	"   distortedTexCoord = osg_MultiTexCoord0;\n"
	"}\n";

static char lensFragmentShader[] =
	VERSION_COMMON
	FRAGMENT_COMMON
	FRAGMENT_HIGHP
	"uniform sampler2D baseTexture1;\n"
	"uniform sampler2D baseTexture2;\n"
	"uniform sampler2D depthTexture1;\n"
	"uniform sampler2D depthTexture2;\n"
	"uniform sampler2D vignetTexture;\n"
	"varying vec4 distortedTexCoord;\n"

	"uniform float inputVect[36];\n"

	"float \n"
	"	render_depth, seed, wx, wy, wz, vx, vy, vz,\n"
	"		k1, k2, t1, t2, k3, k4, k5, k6,\n"
	"			vignet_thresh1, vignet_thresh2, \n"
	"				td, tr, te, ti,\n"
	"				width, height,\n"
	"					fx, fy, ox, oy, extra_margin, extra_zoom,\n"
	"						noise_amplitude, day_light;\n"
	"int extra_sampling_points, double_input;\n"

	//direct sampling = 4, interpolated = 2, + xx + xx + xx +, equals 10 points
	"#define DIRECT_SAMPLING_POINTS 2\n" //for motion blur
	"#define MAX_LENS_ITERATIONS (int(inputVect[8]))\n" //points interploated between two sampling points, more efficient than just using DIRECT_SAMPLING_POINTS
	"#define MAX_PIX_ERR inputVect[9]\n"

	"#define aa 8121.	\n"
	"#define cc 0.845213304	\n"
	"#define rand(seed) fract(aa*seed+cc)\n" //inputs and outputs[0.,1.)

	"#define pi 3.141592653\n"

	//ref: https://thebookofshaders.com/edit.php?log=161119150756
	//http://byteblacksmith.com/improvements-to-the-canonical-one-liner-glsl-rand-for-opengl-es-2-0/
	"float random (in vec2 st) {\n"
	"	//return fract(sin(dot(st.xy,vec2(12.9898,78.233))) * 43758.5453);\n"
	"	return fract(sin(mod(dot(st.xy,vec2(12.9898,78.233)),3.14)) * 43758.5453);\n"
	"}\n"

	/* normal random variate generator */
	/* The polar form of the Box-Muller transformation*/
	/* mean 0., standard deviation 1. */
	/* Improvements can be done by getting 2 random number instead of one*/
	"float randomN(in vec2 st)\n"
	"{\n"
	"	float x1, x2, w;\n"
	"	do {\n"
	"		x1 = 2.*random(st) - 1.;\n"
	"		st.x += 0.1;\n"
	"		x2 = 2.*random(st) - 1.;\n"
	"		st.y += 0.1;\n"
	"		w = x1*x1 + x2*x2;\n"
	"	} while (w >= 1.);\n"

    	"	if(w < 0.00001)  w = 0.00001;\n" //prevent w==0 which causes division by zero 
	"	w = sqrt( -(2.*log(w)) / w );\n"
	"	return x1*w; // or x2*w\n"
	"}\n"

	///Inverse Lens///////////////////////////////////////////////////////
	//Distortion based on five parameters correction model used in opencv
	//Here we need to distort but because we use a fragment shader, inverse formula is needed
	//which is the distortion correction(undistortion) model
	//There is no undistortion formula so we use an iterative approach to calc the inverse

	//note: Radial distortion can be written as additive, so that radial and tangential distortion turn can be changed
	//mistake notes: in shelley r:=r2 so dr:= 1+k1*r+k2*r2+k3*r3
	//mistake notes: in OriellyOpenCV, distortion formula is wrong, see other opencv docs
	//mistake notes: in shelley2014(p.33) && opencv doxygen document, distortion formula is used as camera model,
	//mistake notes: opencv tutorial(not doxygen) is !!!WRONG!!! expressing an undistortion model,
	//see:http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
	//NOTICE: this function is different to standard function in that it also removes ox, oy offsets
	"float inverse_lens(in vec2 distortedTexCoord, out vec2 texCoord_noLens) {\n"
	"	int i;\n"
	"	float x, y, u_distorted, v_distorted, u, v, u2, v2, uv, r2, r4, r6, _dr, pix_x, pix_y, error;\n"
	"	vec2 dt, distortion, uv_nolens, pix, res;\n"

		//convert to pixels
	"	x = distortedTexCoord.x*width;\n"
	"	y = (1.-distortedTexCoord.y)*height;\n" //y in a picture is defined from up to down

	"	u_distorted = (x - ox)/fx;\n"
	"	v_distorted = (y - oy)/fy;\n"

		//First assumption: dt, dr does not change much after distortion
		//By iterating we reach real coordinates
	"	u = u_distorted;\n" //first assumption
	"	v = v_distorted;\n"

	"	for(i = 0; i < MAX_LENS_ITERATIONS; i++) {\n"
	"		u2 = u*u; v2 = v*v; uv = u*v;\n"
	"		r2 = u2 + v2; r4 = r2*r2; r6 = r4*r2;\n"
	"		dt = vec2(2.*t1*uv + t2*(r2+2.*u2),"
	"				t1*(r2+2.*v2) + 2.*t2*uv);\n"
	"		_dr = 1. + k1*r2 + k2*r4 + k3*r6;\n" //3 param radial distortion
	//"		_dr = (1. + k1*r2 + k2*r4 + k3*r6)/(1. + k4*r2 + k5*r4 + k6*r6);\n" //6 param radial distortion
    
            //Increases fps
    "       //res = vec2(u_distorted, v_distorted) - (_dr* vec2(u, v) + dt);\n"
    "       //if(abs(res.x*width*1.42) < MAX_PIX_ERR && abs(res.y*height*1.42) < MAX_PIX_ERR)\n"
    "       //     break;\n"

            //Consider convergance to out of ideal(bigger) frame as divergance
            //Increases fps
    "       pix_x = u/extra_margin*fx+ox; pix_y = v/extra_margin*fy+oy;\n"
    "       if(pix_x <0. || pix_x > width || pix_y < 0. || pix_y > height)\n"
    "            return 1000.;\n" //high value as error
    
	"		distortion = (_dr*vec2(u, v) + dt) - vec2(u, v);\n"
	"		u = u_distorted - distortion.x;\n"
	"		v = v_distorted - distortion.y;\n"
	"	}\n"

	"	uv_nolens = vec2(u, v);\n"
	"	uv_nolens *= extra_zoom;\n"

		//convert to ideal rendered coordinates, in ideal coordinates, ox, oy offsets are not used
	"	pix = vec2(fx*uv_nolens.x + width/2., fy*uv_nolens.y + height/2.);\n" //converting to pixels in out texture
	"	texCoord_noLens = vec2(pix.x/width, 1.-pix.y/height);\n" //texCoord is defined from up to down

        //final error calculation
    "	u2 = u*u; v2 = v*v; uv = u*v;\n"
	"	r2 = u2 + v2; r4 = r2*r2; r6 = r4*r2;\n"
    "   res = vec2(u_distorted, v_distorted) - (_dr* vec2(u, v) + dt);\n"
    "   res.x*=width; res.y*=height;\n"
	"   error = length(res);\n"
    "   return error; //error in pixels \n"
	"}\n"
	
	///Inverse Rolling Shutter////////////////////////////////////////////////////////////
	//differentiating camera model h(shelley-eq5.4)) with respect to X, Y, Z
	//dh/dt = dh/dX*dX/dt + dh/dY*dY/dt + dh/dZ*dZ/dt
	//calculating fixed points speed in cam coordinates
	//eq.2-102, Greenwood, principles of Dynamics, 2nd edition, p.50
	//Vel := vec3(dX/dt, dY/dt, dZ/dt) = velocity with respect to cam axis, in cam axis
	//Vel = speed of camera (wrt inertia in cam axes) + ...
	// ... + speed of point (wrt. inertia in in cam axes) + ...
	// ... + angular speed of camera (in cam axes) x ...
	// ... x position of feature(in cam axes)
	//Vel = VCamWrtInertiaInCam + VpointWrtInertiaInCam + WCamWrtInertiaInCam x RpointInCam
	//			= (CNED2Cam * VCamWrtInertiaInNed) + (CNED2Cam * VpointWrtInertiaInNed) + (CNED2Cam * WCamWrtInertiaInNed) x RpointInCam
	//Rpoint = vec3(X, Y, Z) //point pose in cam
	//Vel = -VCam - cross(WCam, Rpoint) //velocity of point in cam axis
	//Vel2Z = -VCam2Z -cross(WCam, Rpoint2Z)
	//udot = Vel2Z.x - u*Vel2Z.z
	//vdot = Vel2Z.y - v*Vel2Z.z

	//When no lens distortion:
	//dh2du = vec2(fx, 0)
	//dh2dv = vec2(0, fy)

	//dh2dt = dh2du*udot + dh2dv*vdot = vec2(fx*udot, fy*vdot);\n"

	//h_with_rs(pixels) = h_no_rs(pixels) + dh2dt*deltaT; //dh2dt calculated at h_no_rs
	//or: h_no_rs(pixels) = h_with_rs(pixels) - dh2dt*deltaT; //dh2dt calculated at h_with_rs

	//Assuming no u, v change in RS length(so no udot, vdot change)
	"void calc_dh2dt(in vec2 h, in vec3 _VCam2Z, in vec3 _WCam, out vec2 _dh2dt) {"
	"	float u, v, udot, vdot;\n"
	"	vec3 Rpoint2Z, Vel2Z;\n"
	"	u = (h.x - width/2.)/fx;\n"
	"	v = (h.y - height/2.)/fy;\n"
	"	Rpoint2Z = vec3(u, v, 1.);\n" //point pose in cam / Z
	"	Vel2Z = -_VCam2Z -cross(_WCam, Rpoint2Z);\n"
	"	udot = Vel2Z.x - u*Vel2Z.z;\n"
	"	vdot = Vel2Z.y - v*Vel2Z.z;\n"
		//We are after lens, so we assume no lens distortion
		//Lens just affects deltaT(pixel syncronous lines occur after lens) which we computed before
	"	_dh2dt = vec2(fx*udot, fy*vdot);\n" //calculated at h_no_rs(1st approximation)
	"}\n"

	"void inverse_rolling_shutter(in vec2 distortedTexCoord_, in vec2 texCoord_noLens, in float frame_time_offset, in int frame_num, out vec2 texCoord_noLensNoRS[DIRECT_SAMPLING_POINTS]) {\n"
	"	int i,j,row, RK2_STEPS;\n"
	"	float deltaT, deltaT2, Z;\n"
	"	vec2 h_no_rs, h_with_rs, h1, h2, dh2dt0, dh2dt1, dh2dt2, texCoord_realRes;\n"
	"	vec3 VCam2Z, WCam, Vel2Z;\n"
	"	vec4 depthTex;\n"

		//positive deltaT means future
		//we assume upper rows are scanned first(past, older) so have negetive deltaT
	"	//row = distortedTexCoord_.x*width;\n"
	"	row = int((1.-distortedTexCoord_.y)*height);\n"

	"	deltaT = -frame_time_offset - td + tr*(0.5-distortedTexCoord_.y) + (row/2*2==row?-ti:0.);\n" // distortedTexCoord increases from down to up and left to right

		//for comparison
	"	//if(texCoord_noLens.x < 0.5) {deltaT = 0.;texCoord_noLens.x+=0.5;}\n"

	"	WCam = vec3(wx, wy, wz);\n" //W of cam/ned in cam axes

		//Assuming no depth change in RS period
	"	texCoord_realRes = (texCoord_noLens - vec2(.5, .5)) / extra_margin + vec2(.5, .5);\n" //zoom to needed resolution
	"	if(frame_num == 0)\n"
	"		depthTex = texture2D (depthTexture1, texCoord_realRes);\n" //no space after texture2D will cause warning
	"	else\n"
	"		depthTex = texture2D (depthTexture2, texCoord_realRes);\n" //no space after texture2D will cause warning

	"	Z = depthTex.r*(255.*255.) + depthTex.g*255. + depthTex.b\n;" //verified
	"	//Z = 3000.;\n"

	"	VCam2Z = vec3(vx / Z, vy / Z, vz / Z);\n" //Vel of aircraft wrt. ned in camera coordinates

		//Corrected texCoord can be converted to u, v with a linear relationship
	"	h_with_rs = vec2(texCoord_noLens.x*width, (1.-texCoord_noLens.y)*height);\n"
	"	calc_dh2dt(h_with_rs, VCam2Z, WCam, dh2dt0);\n" //calculated at h_with_rs

	"	for(i = 0; i < DIRECT_SAMPLING_POINTS; i++) {\n"
	"		deltaT2 = deltaT;\n"
	"		if(DIRECT_SAMPLING_POINTS > 1) \n"
	"			deltaT2 += te*(-0.5 + float(i)/float(DIRECT_SAMPLING_POINTS-1));\n" //from te*(-0.5)(past) to te*(+0.5)(future)

			//calculating h_no_rs
			// RK2 pixel errors based on degree of predicted rotations
			// every2deg->0.02pix, 6-> 0.01, 12->0.11, 30->1.65
	"		//RK2_STEPS = 2;\n"
			//experimentally add a step at least every 2.5 degrees of rotation
	"		RK2_STEPS = int(abs(deltaT2*length(WCam))*180./pi/2.5) + 1;\n"
	"		deltaT2 /= float(RK2_STEPS);\n"

	"		for(j=0;j<RK2_STEPS;j++){\n"
	"			if(j==0) {\n"
	"				h1 = h_with_rs;\n"
	"				dh2dt1 = dh2dt0;\n"
	"			} else {\n"
	"				h1 = h2;\n"
	"				calc_dh2dt(h1, VCam2Z, WCam, dh2dt1);\n"
	"			}\n"

	"			h2 = h1 - dh2dt1*deltaT2;\n" //first assumption (RK1 solution)
				//recalculation of derivative at second point(1st approximation)
	"			calc_dh2dt(h2, VCam2Z, WCam, dh2dt2);\n"

				//RK2 solution
	"			h2 = h1 - (dh2dt1 + dh2dt2)*0.5*deltaT2;\n"
	"		}\n"

	"		h_no_rs = h2;\n"
	"		texCoord_noLensNoRS[i] = vec2(h_no_rs.x/width, 1.-h_no_rs.y/height);"
	"	}\n"
	"}\n"

	///Motion Blur////////////////////////////////////////////////////////
	//also zooms to center rectangle
	"void blur_distortion(in vec2 texCoord_corrected[DIRECT_SAMPLING_POINTS], in int frame_num, out vec4 texColor) \n"
	"{	\n"
	"	vec2 texCoord3_interpolated, texCoord_realRes;\n"
	"	vec4 color;\n"
	"	int i, j, count, do_break;\n"
	"	texColor = vec4(0., 0., 0., 0.);\n"
	"	count = 0;\n"
	"	do_break = 0;\n"
	"	for(i = 0; i < DIRECT_SAMPLING_POINTS; i++) {\n"
	"		for(j = 0; j < 1+extra_sampling_points; j++) {\n"
	"			count++;\n"
	"			if(i != DIRECT_SAMPLING_POINTS - 1)\n" //not last
	"				texCoord3_interpolated = (texCoord_corrected[i+1].xy*float(j) + texCoord_corrected[i].xy*float(extra_sampling_points + 1 - j))/float(extra_sampling_points + 1);\n"
	"			else \n" //i is the last direct sampling point
	"				texCoord3_interpolated = texCoord_corrected[i].xy;\n"

				//convert to real resolution
	"			texCoord_realRes = (texCoord3_interpolated - vec2(.5, .5)) / extra_margin + vec2(.5, .5);\n" //zoom to needed resolution
	"			if(texCoord_realRes.x <= 1. && texCoord_realRes.x >= 0. && texCoord_realRes.y <= 1. && texCoord_realRes.y >= 0.){\n"		
	"				if(frame_num==0)\n"
	"					color = texture2D (baseTexture1, texCoord_realRes.xy);\n" //no space after texture2D will cause warning
	"				else\n"
	"					color = texture2D (baseTexture2, texCoord_realRes.xy);\n" //no space after texture2D will cause warning
	"				if(color.a < 0.5)\n" //SKY
	"					texColor+=vec4(0.,0.749,1., 1.);\n"
	"				else\n"
	"					texColor+=color;\n"

	"			} else {\n" //out of range
	"				texColor = vec4(0., 0., 0., 0.);\n"
	"				do_break = 1; break;\n"
	"			}\n"
	"			if(i == DIRECT_SAMPLING_POINTS - 1) {do_break = 1; break;}\n" //don't extrapolate after last sampling point
	"		}\n"
	"		if(do_break == 1) break;\n"
	"	}\n"
	"	texColor /= float(count);\n" //sampling = 4, interpolated = 2, + xx + xx + xx +, equals 10 points
	"}\n"

	//Main////////////////////////////////////////////////////////
	"void main() {\n"
	"	float n1,n2,n3,s1,s2, err;\n"
	"	vec2 texCoord_noLens, texCoord3f1[DIRECT_SAMPLING_POINTS], texCoord3f2[DIRECT_SAMPLING_POINTS], texCoord_realRes, st;\n"
	"	vec2 mean_texCoord, texCoord_realResMean, texCoord_realResMin, texCoord_realResMax;\n"
	"	vec4 texColor, vignetTex, depthTex;\n"

		///Getting uniform inputs
	"	render_depth = inputVect[0];\n"
	"	seed = inputVect[1];\n"
	"	wx = inputVect[2]; wy = inputVect[3]; wz = inputVect[4];\n" //W of cam/ned in cam axes
	"	vx = inputVect[5]; vy = inputVect[6]; vz = inputVect[7];\n" //Vel of aircraft wrt. ned in camera coordinates
	"	k1 = inputVect[10]; k2 = inputVect[11]; t1 = inputVect[12]; t2 = inputVect[13]; k3 = inputVect[14]; k4 = inputVect[15]; k5 = inputVect[16]; k6 = inputVect[17];\n"
	"	vignet_thresh1 = inputVect[18]; vignet_thresh2 = inputVect[19];\n"
	"	td = inputVect[20];\n" //center row delay of camera
	"	tr = inputVect[21];\n" //up2down delay of camera, sign verified
	"	te = inputVect[22];\n" //in-row delay, works when DIRECT_SAMPLING_POINTS > 1
	"	extra_sampling_points = int(inputVect[23]);\n" //points interpolated between two sampling points, more efficient than just using DIRECT_SAMPLING_POINTS
	"	ti = inputVect[24];\n"
	"	width = inputVect[25]; height = inputVect[26];\n" //for normalizing calibration params
	"	fx = inputVect[27]; fy = inputVect[28];\n" //width, height, causes changing t1 = inputVect[10] if used by #define!!!!
	"	ox = inputVect[29]; oy = inputVect[30];\n"
	"	extra_margin = inputVect[31];\n" //extra margin used in ideally rendered image
	"	extra_zoom = inputVect[32];\n" //used so that image get smaller
	"	day_light = inputVect[33]; noise_amplitude = inputVect[34];\n"
	"	double_input = int(inputVect[35]);\n"

        ///Photometric distortion
	"	vignetTex = texture2D (vignetTexture, distortedTexCoord.xy);\n" //no space after texture2D will cause warning
	"	vignetTex = vec4(vignetTex.x>vignet_thresh2?1.:vignetTex.x, vignetTex.y>vignet_thresh2?1.:vignetTex.y, vignetTex.z>vignet_thresh2?1.:vignetTex.z, 1.);\n"
	"	vignetTex = vec4(vignetTex.x<vignet_thresh1?0.:vignetTex.x, vignetTex.y<vignet_thresh1?0.:vignetTex.y, vignetTex.z<vignet_thresh1?0.:vignetTex.z, 1.);\n"

    "   n1=0.;n2=0.;n3=0.;\n"
    "	if(render_depth < 0.5) {\n" //render color
	"		st = distortedTexCoord.xy;\n"
	"		s1 = rand(fract(seed));\n"
	"		st.x+=s1;\n"
			//uniform noise
	"		//n1 = random(st) - 0.5;st.y+=s1;\n" //red
	"		//n2 = random(st) - 0.5;st.x-=s1;\n" //green
	"		//n3 = random(st) - 0.5;\n" //blue

			//Gaussian noise
	"		n1 = noise_amplitude*randomN(st);st.y+=s1;\n" //red
	"		n2 = noise_amplitude*randomN(st);st.x-=s1;\n" //green
	"		n3 = noise_amplitude*randomN(st);\n" //blue    
    "   }\n"
    
        //vignet invisible, no need for further processing
    "   if(vignetTex.x < 0.001 && vignetTex.y < 0.001 && vignetTex.z < 0.001) {\n" 
    "       mgl_FragColor = vec4(n1, n2, n3, 1.);\n"
    "       return;\n"
    "   }\n"
        
		///Geometric distortion
		//In fragment shader, movement is done from final pixel coordinates to ideal input pixel coordinates
		//Here we reach from distorted to undistorted coordinates
	"	err = inverse_lens(distortedTexCoord.xy, texCoord_noLens);\n" //classic
	"	//texCoord_noLens = distortedTexCoord.xy;\n"
        //Pure red as Divergance Warning
    "   if(err > MAX_PIX_ERR) {\n"
    "       mgl_FragColor = vec4(1., 0., 0., 1.);\n"
    "       return;\n"
    "   }\n"
    
		///Temporal distortion
	"	//texCoord3f1[0] = texCoord_noLens;\n"
	"	if(double_input == 0)\n"
	"		inverse_rolling_shutter(distortedTexCoord.xy, texCoord_noLens, -td-0.5*ti          , 0, texCoord3f1);\n"
	"	else {\n"
	"		inverse_rolling_shutter(distortedTexCoord.xy, texCoord_noLens, -td-0.5*tr-ti-0.5*te, 0, texCoord3f1);\n"
	"		inverse_rolling_shutter(distortedTexCoord.xy, texCoord_noLens, -td+0.5*tr-0.+0.5*te, 1, texCoord3f2);\n"
	"	}\n"

	"	//if(texCoord_noLens.x > 0.5) {\n" //render depth, not color
	"	if(render_depth > 0.5) {\n" //render depth, not color
	"		mean_texCoord = (texCoord3f1[0] + texCoord3f1[DIRECT_SAMPLING_POINTS-1])/2.;\n"
	"		texCoord_realResMean = (mean_texCoord - vec2(.5, .5)) / extra_margin + vec2(.5, .5);\n"
	"		texCoord_realResMin = (texCoord3f1[0] - vec2(.5, .5)) / extra_margin + vec2(.5, .5);\n"
	"		texCoord_realResMax = (texCoord3f1[DIRECT_SAMPLING_POINTS-1] - vec2(.5, .5)) / extra_margin + vec2(.5, .5);\n"
	"		if(texCoord_realResMin.x > 0. && texCoord_realResMin.x < 1. && texCoord_realResMin.y > 0. && texCoord_realResMin.y < 1. &&  \n"
	"			texCoord_realResMax.x > 0. && texCoord_realResMax.x < 1. && texCoord_realResMax.y > 0. && texCoord_realResMax.y < 1.) {  \n"
	"			depthTex = texture2D (depthTexture1, texCoord_realResMean);\n" //no space after texture2D will cause warning\n"
	"		} else {\n"
	"			depthTex = vec4(1., 0., 0., 1.);\n"
	"		}\n"
	"		mgl_FragColor = depthTex;\n"
	"	} else {\n" //render color

			///Temporal distortion, averaging part
	"		//if(texCoord_noLens.x < 0.5) \n"
	"			blur_distortion(texCoord3f1, 0, texColor);\n"
	"		//else \n"
	"		//	blur_distortion(texCoord3f2, 1, texColor);\n"

	"		mgl_FragColor = vec4(vignetTex.x*day_light*texColor.r + n1,"
	"								vignetTex.y*day_light*texColor.g + n2,"
	"									vignetTex.z*day_light*texColor.b + n3, 1.);\n"

	"		//if(texCoord_noLens.x < 0.47)\n"
	"		//	mgl_FragColor = texture2D (baseTexture1, texCoord_noLens.xy);\n" //no space after texture2D will cause warning
	"		//else if(texCoord_noLens.x < 0.5)\n"
	"		//	mgl_FragColor = texture2D (depthTexture1, texCoord_noLens.xy);\n" //no space after texture2D will cause warning
	"		//else if(texCoord_noLens.x < 0.53)\n"
	"		//	mgl_FragColor = texture2D (baseTexture2, texCoord_noLens.xy);\n" //no space after texture2D will cause warning
	"		//else if(texCoord_noLens.x < 0.57)\n"
	"		//	mgl_FragColor = texture2D (depthTexture2, texCoord_noLens.xy);\n" //no space after texture2D will cause warning
	"	}\n"
	"}\n";
