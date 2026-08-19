// Standalone check of the bad_orientation logic. The deploy binary needs unitree_sdk2 +
// DDS + cnpy and only builds on the robot, so the guard logic is lifted verbatim here and
// exercised against the cases that matter: upright, tipped, and unreadable IMU.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

inline constexpr int kBadOrientationHoldChecks = 3;

static bool bad_orientation(const float* data, float limit_angle = 1.0)
{
    const float norm = std::sqrt(data[0] * data[0] + data[1] * data[1] + data[2] * data[2]);
    if (!std::isfinite(norm) || std::fabs(norm - 1.0f) > 0.1f)
    {
        return false;
    }
    const float tilt = std::acos(std::clamp(-data[2] / norm, -1.0f, 1.0f));
    static int consecutive = 0;
    consecutive = (tilt > limit_angle) ? consecutive + 1 : 0;
    return consecutive >= kBadOrientationHoldChecks;
}

static void gravity_at(float tilt_deg, float* out)
{
    const float r = tilt_deg * M_PI / 180.0f;   // tilt about X: g rotates into +y
    out[0] = 0.0f; out[1] = std::sin(r); out[2] = -std::cos(r);
}

static int failures = 0;
static void expect(const char* what, bool got, bool want)
{
    const bool ok = got == want;
    if (!ok) ++failures;
    std::printf("  [%s] %-52s got=%d want=%d\n", ok ? "ok" : "FAIL", what, got, want);
}

// Feed the same reading N times; returns whether it trips within the hold window.
static bool sustained(const float* g, int n = kBadOrientationHoldChecks)
{
    bool tripped = false;
    for (int i = 0; i < n; ++i) tripped = bad_orientation(g);
    return tripped;
}

int main()
{
    float g[3];
    const float zero[3] = {0.0f, 0.0f, 0.0f};
    const float nan_read[3] = {NAN, NAN, NAN};
    const float unnormalised[3] = {0.0f, 0.0f, -9.81f};   // m/s^2 instead of a unit vector

    gravity_at(0.0f, g);   expect("upright (0 deg) does not trip", sustained(g), false);
    gravity_at(45.0f, g);  expect("45 deg, under the 57 deg limit, does not trip", sustained(g), false);
    gravity_at(70.0f, g);  expect("70 deg, over the limit, trips when sustained", sustained(g), true);
    gravity_at(180.0f, g); expect("fully inverted trips", sustained(g), true);

    expect("all-zero reading is treated as no reading", sustained(zero), false);
    expect("NaN reading is treated as no reading", sustained(nan_read), false);
    expect("un-normalised (m/s^2) reading is rejected", sustained(unnormalised), false);

    // Debounce: one bad frame in the middle of good ones must not trip.
    gravity_at(0.0f, g); sustained(g);
    float bad[3]; gravity_at(90.0f, bad);
    bool tripped = bad_orientation(bad);
    gravity_at(0.0f, g);
    tripped = bad_orientation(g) || tripped;
    expect("a single bad frame between good ones does not trip", tripped, false);

    // And a real tip-over does trip, only after the hold window.
    gravity_at(0.0f, g); sustained(g);
    gravity_at(80.0f, bad);
    expect("first over-limit check alone does not trip", bad_orientation(bad), false);
    expect("second does not trip either", bad_orientation(bad), false);
    expect("third trips", bad_orientation(bad), true);

    std::printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "all cases pass",
                failures, failures == 1 ? "" : "s");
    return failures != 0;
}
