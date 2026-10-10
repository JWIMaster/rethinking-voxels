// Horizontal-only KinoStreak pyramid, packed side by side in a half-height atlas.
// Level 0 is half width; stop once the smallest level reaches <= 16 pixels.
// Ten levels cover render widths up to 16384 without changing vertical resolution.
int StreakWidth(int fullWidth, int level) {
    return max(1, fullWidth >> (level + 1));
}

int StreakOffset(int fullWidth, int level) {
    int offset = 0;
    for (int i = 0; i < level; ++i) offset += StreakWidth(fullWidth, i);
    return offset;
}

int StreakLastLevel(int fullWidth) {
    int level = 0;
    while (level < 9 && StreakWidth(fullWidth, level) > 16) ++level;
    return level;
}
