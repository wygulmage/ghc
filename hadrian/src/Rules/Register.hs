module Rules.Register (
    configurePackageRules, registerPackageRules, registerPackages,
    libraryTargets
    ) where

import Base
import Context
import Expression ( getContextData )
import Hadrian.BuildPath
import Hadrian.Expression
import Hadrian.Haskell.Cabal
import Oracles.Setting
import Packages
import Rules.Gmp
import Rules.Rts
import Settings
import Target
import Utilities

import Hadrian.Haskell.Cabal.Type
import qualified Text.Parsec      as Parsec

import Distribution.Version (Version)
import qualified Distribution.Parsec as Cabal
import qualified Distribution.Types.PackageName as Cabal
import qualified Distribution.Types.PackageId as Cabal

import qualified Hadrian.Haskell.Cabal.Parse as Cabal
import qualified System.Directory            as IO

-- * Configuring

-- | Configure a package and build its @setup-config@ file, as well as files in
-- the @build/pkgName/build/autogen@ directory.
configurePackageRules :: Rules ()
configurePackageRules = do
    root <- buildRootRules
    root -/- "**/setup-config" %> \out -> do
        (stage, path) <- parsePath (parseSetupConfig root) "<setup config path parser>" out
        let pkg = unsafeFindPackageByPath path
        Cabal.configurePackage (Context stage pkg vanilla)

    root -/- "**/autogen/cabal_macros.h" %> \out -> do
        (stage, path) <- parsePath (parseToBuildSubdirectory root) "<cabal macros path parser>" out
        let pkg = unsafeFindPackageByPath path
        Cabal.buildAutogenFiles (Context stage pkg vanilla)

    root -/- "**/autogen/Paths_*.hs" %> \out ->
        need [takeDirectory out -/- "cabal_macros.h"]

parseSetupConfig :: FilePath -> Parsec.Parsec String () (Stage, FilePath)
parseSetupConfig root = do
    _ <- Parsec.string root *> Parsec.optional (Parsec.char '/')
    stage <- parseStage
    _ <- Parsec.char '/'
    pkgPath <- Parsec.manyTill Parsec.anyChar
        (Parsec.try $ Parsec.string "/setup-config")
    return (stage, pkgPath)

parseToBuildSubdirectory :: FilePath -> Parsec.Parsec String () (Stage, FilePath)
parseToBuildSubdirectory root = do
    _ <- Parsec.string root *> Parsec.optional (Parsec.char '/')
    stage <- parseStage
    _ <- Parsec.char '/'
    pkgPath <- Parsec.manyTill Parsec.anyChar
        (Parsec.try $ Parsec.string "/build/")
    return (stage, pkgPath)

-- * Registering

registerPackages :: [Context] -> Action ()
registerPackages ctxs = do
    need =<< mapM pkgRegisteredLibraryFile ctxs

    -- Dynamic RTS library files need symlinks (Rules.Rts.rtsRules).
    forM_ ctxs $ \ ctx -> when (package ctx == rts) $ do
        ways <- interpretInContext ctx (getLibraryWays <> getRtsWays)
        needRtsSymLinks (stage ctx) ways

-- | Register a package and initialise the corresponding package database if
-- need be. Note that we only register packages in 'Stage0' and 'Stage1'.
registerPackageRules :: [(Resource, Int)] -> Stage -> Rules ()
registerPackageRules rs stage = do
    root <- buildRootRules

    -- Initialise the package database.
    root -/- relativePackageDbPath stage -/- packageDbStamp %> \stamp ->
        writeFileLines stamp []

    -- Register a package.
    root -/- relativePackageDbPath stage -/- "*.conf" %> \conf -> do
        historyDisable
        let libpath = takeDirectory (takeDirectory conf)
            settings = libpath -/- "settings"
            platformConstants = libpath -/- "platformConstants"

        need [settings, platformConstants]

        pkgName <- getPackageNameFromConfFile conf
        let pkg = unsafeFindPackageByName pkgName
        isBoot <- (pkg `notElem`) <$> stagePackages Stage0

        let ctx = Context stage pkg vanilla
        case stage of
            Stage0 | isBoot -> copyConf  rs ctx conf
            _               -> buildConf rs ctx conf

buildConf :: [(Resource, Int)] -> Context -> FilePath -> Action ()
buildConf _ context@Context {..} conf = do
    depPkgIds <- cabalDependencies context
    ensureConfigured context
    need =<< mapM (\pkgId -> packageDbPath stage <&> (-/- pkgId <.> "conf")) depPkgIds

    ways <- interpretInContext context (getLibraryWays <> if package == rts then getRtsWays else mempty)
    need =<< concatMapM (libraryTargets True) [ context { way = w } | w <- ways ]

    -- We might need some package-db resource to limit read/write, see packageRules.
    path <- buildPath context

    -- Special package cases (these should ideally be rolled into Cabal).
    when (package == rts) $
        -- If Cabal knew about "generated-headers", we could read them from the
        -- 'configuredCabal' information, and just "need" them here.
        need [ path -/- "DerivedConstants.h"
             , path -/- "ghcautoconf.h"
             , path -/- "ghcplatform.h"
             , path -/- "ghcversion.h" ]

    when (package == integerGmp) $ need [path -/- gmpLibraryH]

    -- Copy and register the package.
    Cabal.copyPackage context
    Cabal.registerPackage context

    -- The above two steps produce an entry in the package database, with copies
    -- of many of the files we have build, e.g. Haskell interface files. We need
    -- to record this side effect so that Shake can cache these files too.
    -- See why we need 'fixWindows': https://gitlab.haskell.org/ghc/ghc/issues/16073
    let fixWindows path = do
            version  <- setting GhcVersion
            hostOs   <- cabalOsString <$> setting BuildOs
            hostArch <- cabalArchString <$> setting BuildArch
            let dir = hostArch ++ "-" ++ hostOs ++ "-ghc-" ++ version
            return $ if windowsHost then path -/- "../.." -/- dir else path
    pkgDbPath <- fixWindows =<< packageDbPath stage
    let dir = pkgDbPath -/- takeBaseName conf
    files <- liftIO $ getDirectoryFilesIO "." [dir -/- "**"]
    produces files

copyConf :: [(Resource, Int)] -> Context -> FilePath -> Action ()
copyConf rs context@Context {..} conf = do
    depPkgIds <- fmap stdOutToPkgIds . askWithResources rs $
        target context (GhcPkg Dependencies stage) [pkgName package] []
    need =<< mapM (\pkgId -> packageDbPath stage <&> (-/- pkgId <.> "conf")) depPkgIds
    -- We should unregister if the file exists since @ghc-pkg@ will complain
    -- about existing package: https://github.com/snowleopard/hadrian/issues/543.
    -- Also, we don't always do the unregistration + registration to avoid
    -- repeated work after a full build.
    -- We do not track 'doesFileExist' since we are going to create the file if
    -- it is currently missing. TODO: Is this the right thing to do?
    -- See https://github.com/snowleopard/hadrian/issues/569.
    unlessM (liftIO $ IO.doesFileExist conf) $ do
        buildWithResources rs $
            target context (GhcPkg Unregister stage) [pkgName package] []
        buildWithResources rs $
            target context (GhcPkg Copy stage) [pkgName package] [conf]
  where
    stdOutToPkgIds :: String -> [String]
    stdOutToPkgIds = drop 1 . concatMap words . lines

getPackageNameFromConfFile :: FilePath -> Action String
getPackageNameFromConfFile conf
    | takeBaseName conf == "rts" = return "rts"
    | otherwise = case parseCabalName (takeBaseName conf) of
        Left err -> error $ "getPackageNameFromConfFile: Couldn't parse " ++
                            takeBaseName conf ++ ": " ++ err
        Right (name, _) -> return name

parseCabalName :: String -> Either String (String, Version)
parseCabalName = fmap f . Cabal.eitherParsec
  where
    f :: Cabal.PackageId -> (String, Version)
    f pkg_id = (Cabal.unPackageName $ Cabal.pkgName pkg_id, Cabal.pkgVersion pkg_id)

-- | Return extra library targets.
extraTargets :: Context -> Action [FilePath]
extraTargets context
    | package context == rts  = needRtsLibffiTargets (Context.stage context)
    | otherwise               = return []

-- | Given a library 'Package' this action computes all of its targets. Needing
-- all the targets should build the library such that it is ready to be
-- registered into the package database.
-- See 'packageTargets' for the explanation of the @includeGhciLib@ parameter.
libraryTargets :: Bool -> Context -> Action [FilePath]
libraryTargets includeGhciLib context@Context {..} = do
    libFile  <- pkgLibraryFile     context
    ghciLib  <- pkgGhciLibraryFile context
    ghci     <- if includeGhciLib && not (wayUnit Dynamic way)
                then interpretInContext context $ getContextData buildGhciLib
                else return False
    extra    <- extraTargets context
    return $ [ libFile ]
          ++ [ ghciLib | ghci ]
          ++ extra
