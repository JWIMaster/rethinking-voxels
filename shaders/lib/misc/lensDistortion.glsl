//Barrel Distortion//
#if BARREL_DISTORTION > 0 && defined FRAGMENT_SHADER
    // Post process barrel distortion, meant to emulate the curvature of an
    // anamorphic lens. Radial in screen space, so the distortion rings stay
    // circles around the center of the screen at any aspect ratio.
    // The coordinate is scaled by how far the corners are pulled inwards,
    // which keeps the distorted image filling the entire screen.
    vec2 GetBarrelDistortedTexCoord(vec2 coord) {
        vec2 p = (coord - 0.5) * 2.0 * vec2(aspectRatio, 1.0);

        float r2 = dot(p, p) / (aspectRatio * aspectRatio + 1.0); //0 at the center, 1 in the corners

        vec2 strength = vec2(BARREL_DISTORTION * 0.01);
        #if BARREL_DISTORTION_MODE == 1
            strength.x *= 2.0; //The horizontal element of an anamorphic lens distorts more than the vertical one
        #endif

        p *= (1.0 + strength * r2) / (1.0 + strength);

        return p / vec2(aspectRatio, 1.0) * 0.5 + 0.5;
    }
#endif
