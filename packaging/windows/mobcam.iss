; Inno Setup script for Mob Cam.
; Build with: ISCC /DMyAppVersion=0.1.0 mobcam.iss
; MyAppId is a fixed GUID so upgrades replace the old install instead of side-by-side installing.
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif
#define MyAppName "Mob Cam"
#define MyAppPublisher "Robin Kumar"
#define MyAppExeName "mobcam.exe"
#define MyAppId "{B6E1B6C1-9E1E-4F1B-8B2B-3B9C1A2F5D77}"

[Setup]
AppId={{#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
UninstallDisplayIcon={app}\{#MyAppExeName}
OutputBaseFilename=MobCam-Setup-{#MyAppVersion}
OutputDir=output
Compression=lzma2
SolidCompression=yes
SetupIconFile=mobcam.ico
ArchitecturesInstallIn64BitMode=x64
DisableProgramGroupPage=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional shortcuts:"; Flags: unchecked

[Files]
Source: "..\..\dist\mobcam\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent unchecked

[Code]
// Same rule as the Linux postinst and the macOS in-app help text: never
// install the camera/mic driver automatically, only tell the user how.
procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
  begin
    MsgBox('Mob Cam installed.' + #13#10 +
      'A virtual camera and microphone driver are required and are not installed by this setup.' + #13#10#13#10 +
      'Camera: install OBS Studio and launch it once (this adds the OBS Virtual Camera).' + #13#10 +
      'Microphone: install VB-CABLE, then set Mob Cam''s output to "CABLE Input" - apps record from "CABLE Output".',
      mbInformation, MB_OK);
  end;
end;
