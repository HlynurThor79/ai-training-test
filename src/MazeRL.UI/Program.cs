using Avalonia;

namespace MazeRL.UI;

internal static class Program
{
    [STAThread]
    public static void Main(string[] args) =>
        BuildAvaloniaApp().StartWithClassicDesktopLifetime(args);

    public static AppBuilder BuildAvaloniaApp() =>
        AppBuilder.Configure<App>()
            .UsePlatformDetect()
            // WSLg has no XDG desktop portal (use Avalonia's own file dialogs)
            // and no libICE/libSM (skip X11 session management).
            .With(new X11PlatformOptions
            {
                UseDBusFilePicker = false,
                EnableSessionManagement = false,
            })
            .WithInterFont()
            .LogToTrace();
}
