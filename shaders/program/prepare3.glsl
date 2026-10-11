#include "/lib/common.glsl"

//////Fragment Shader//////Fragment Shader//////
#ifdef FRAGMENT_SHADER
in vec3 dir;

vec2 view = vec2(viewWidth, viewHeight);
vec3 fractCamPos = cameraPositionInt.y == -98257195 ? fract(cameraPosition) : cameraPositionFract;

const ivec2 offsets[8] = ivec2[8](
    ivec2( 1, 0),
    ivec2( 1, 1),
    ivec2( 0, 1),
    ivec2(-1, 1),
    ivec2(-1, 0),
    ivec2(-1,-1),
    ivec2( 0,-1),
    ivec2( 1,-1));

#include "/lib/vx/voxelReading.glsl"
#include "/lib/util/random.glsl"
void main() {
    float dither = nextFloat();
    ivec2 texelCoord = ivec2(gl_FragCoord.xy);
    vec4 writeData = texelFetch(colortex8, texelCoord, 0);
    #ifdef BLOCKLIGHT_HIGHLIGHT
        vec4 writeMatData = texelFetch(colortex3, texelCoord, 0);
    #endif
    if (writeData.a > 1.5) {
        vec4 avgAroundData = vec4(0);
        #ifdef BLOCKLIGHT_HIGHLIGHT
            vec4 avgMatData = vec4(0);
        #endif
        int validAroundCount = 0;
        bool extendable = true;
        for (int k = 0, invalidInARow = 0; k < 10; k++) {
            ivec2 aroundCoord = texelCoord + offsets[k%8];
            if (any(lessThan(aroundCoord, ivec2(0))) || any(greaterThanEqual(aroundCoord, ivec2(view)))) {
                invalidInARow++;
                if (invalidInARow >= 4) { extendable = false; break; }
                continue;
            }
            vec4 aroundData = texelFetch(colortex8, aroundCoord, 0);
            #ifdef BLOCKLIGHT_HIGHLIGHT
                vec4 aroundMatData = texelFetch(colortex3, aroundCoord, 0);
            #endif
            if (aroundData.a > 1.5) {
                invalidInARow++;
                if (invalidInARow >= 4) {
                    extendable = false;
                    break;
                }
                continue;
            }
            invalidInARow = 0;
            if (k < 8) {
                #ifdef BLOCKLIGHT_HIGHLIGHT
                    avgMatData += aroundMatData;
                #endif
                avgAroundData += aroundData;
                validAroundCount++;
            }
        }
        if (extendable && validAroundCount > 0) {
            writeData = avgAroundData / validAroundCount;
            #ifdef BLOCKLIGHT_HIGHLIGHT
                writeMatData = avgMatData / validAroundCount;
            #endif
        } else {
            #ifdef BLOCKLIGHT_HIGHLIGHT
                writeMatData = vec4(0.0);
            #endif
            // fuck view bobbing!
            vec3 rayStartPos = fractCamPos + gbufferModelViewInverse[3].xyz;
            #ifdef PLAYER_VOXELIZATION
                if (firstPersonCamera) {
                    rayStartPos += 0.5 * playerSize * normalize(dir);
                }
            #endif
            vec3 rayHit = rayTrace(rayStartPos, dir, dither);
            float hitDF = getDistanceField(rayHit);
            vec3 gradient = distanceFieldGradient(rayHit);
            float gradientLength2 = dot(gradient, gradient);
            if (hitDF < 0.1 && gradientLength2 > 1e-12 && !any(isnan(gradient)) && !any(isinf(gradient))) {
                writeData.rgb = gradient * inversesqrt(gradientLength2);
                vec4 clipHitPos = gbufferProjection * (gbufferModelView * vec4(rayHit - fractCamPos, 1));
                clipHitPos = 0.5 / clipHitPos.w * clipHitPos + 0.5;
                writeData.a = 1 - clipHitPos.z;
            } else {
                writeData = vec4(0);
            }
        }
    }
    if (any(isnan(writeData)) || any(isinf(writeData))) writeData = vec4(0);
    /*RENDERTARGETS:8*/
    gl_FragData[0] = writeData;
    #ifdef BLOCKLIGHT_HIGHLIGHT
        /*RENDERTARGETS:8,3*/
        gl_FragData[1] = writeMatData;
    #endif
}
#endif

//////Vertex Shader//////Vertex Shader//////
#ifdef VERTEX_SHADER

out vec3 dir;

void main() {
    gl_Position = ftransform();
    dir = normalize((gbufferModelViewInverse * (gbufferProjectionInverse * vec4(gl_Position.xy / gl_Position.w, 0.9999, 1))).xyz) * (voxelVolumeSize.x * 0.4);
}
#endif
